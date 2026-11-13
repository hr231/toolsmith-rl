import http.client
import json
import os
import stat
import urllib.parse
import urllib.request

import pytest

from toolsmith.dataset import SOURCE_PATH, DatasetConfig, generate_dataset, load_records
from toolsmith.episode import Episode
from toolsmith.evaluator import apply_patch
from toolsmith.remote import (
    MAX_BODY_BYTES,
    RemoteClient,
    RemoteProtocolError,
    RemoteSandbox,
    SandboxService,
    ensure_token,
    load_service_records,
    serve,
)
from toolsmith.sandbox import DEFAULT_IMAGE, SandboxError, SandboxLimits

TOKEN = "test-token-0123456789abcdef"
SMALL = DatasetConfig(train=3, dev=3, test_iid=3, test_compositional=2, challenge=4)


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path / "data"
    generate_dataset(SMALL, directory)
    return directory


@pytest.fixture
def server(dataset, tmp_path):
    service = SandboxService(
        load_service_records(dataset),
        token=TOKEN,
        image=os.environ.get("TOOLSMITH_SANDBOX_IMAGE", DEFAULT_IMAGE),
        limits=SandboxLimits(),
        workers=2,
        queue_timeout=0.0,
        score_log=tmp_path / "scores.jsonl",
    )
    httpd = serve(service, port=0)
    yield service, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def raw_post(url, body, token=TOKEN):
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_service_binds_only_to_loopback(server):
    service, url = server
    assert url.startswith("http://127.0.0.1:")


def test_health_requires_the_exact_token(server):
    _, url = server
    assert RemoteClient(url, TOKEN).health()["ok"] is True
    for token in ("wrong", TOKEN + "x", ""):
        with pytest.raises(RemoteProtocolError, match="401"):
            RemoteClient(url, token).health()


def test_unknown_task_and_undeclared_paths_are_rejected(server, dataset):
    _, url = server
    client = RemoteClient(url, TOKEN)
    record = load_records(dataset, "train")[0]
    source = record.task.source_files[0].content
    with pytest.raises(RemoteProtocolError, match="400"):
        client.visible("train-9999-unknown", {SOURCE_PATH: source})
    for files in (
        {},
        {SOURCE_PATH: source, "src/other.py": "x = 1\n"},
        {"tests/test_solution.py": "def test_x():\n    pass\n"},
        {SOURCE_PATH: 42},
    ):
        with pytest.raises(RemoteProtocolError, match="400"):
            client.visible(record.task.task_id, files)


def test_malformed_and_oversized_requests_are_rejected(server, dataset):
    _, url = server
    task_id = load_records(dataset, "train")[0].task.task_id
    duplicate = f'{{"task_id": "{task_id}", "task_id": "{task_id}", "patch": "x"}}'.encode()
    assert raw_post(url + "/v1/score", duplicate)[0] == 400
    assert raw_post(url + "/v1/score", b'{"task_id": NaN, "patch": "x"}')[0] == 400
    assert raw_post(url + "/v1/score", b'{"task_id": "x", "patch": "y", "extra": 1}')[0] == 400
    # Sends only headers: the service must refuse the declared size without reading the body.
    address = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(address.hostname, address.port, timeout=10)
    connection.putrequest("POST", "/v1/score")
    connection.putheader("Authorization", f"Bearer {TOKEN}")
    connection.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
    connection.endheaders()
    assert connection.getresponse().status == 413
    connection.close()
    assert raw_post(url + "/v1/unknown", b"{}")[0] == 404


def test_saturated_service_reports_busy_as_infrastructure_error(server, dataset):
    service, url = server
    record = load_records(dataset, "train")[0]
    held = [service._slots.acquire(blocking=False) for _ in range(2)]
    assert all(held)
    try:
        result = RemoteClient(url, TOKEN).visible(
            record.task.task_id, {SOURCE_PATH: record.task.source_files[0].content}
        )
        assert result.status == "infrastructure_error"
    finally:
        for _ in held:
            service._slots.release()


def test_unreachable_service_ends_episode_as_infrastructure_error(dataset):
    record = load_records(dataset, "train")[0]
    client = RemoteClient("http://127.0.0.1:9", TOKEN, timeout=2)
    with Episode(record.task, sandbox=RemoteSandbox(record.task, client)) as episode:
        event = episode.dispatch("run_visible_tests", {})
    assert event.observation["status"] == "infrastructure_error"
    assert episode.terminal_reason == "infrastructure_error"


def test_episode_rejects_a_sandbox_for_a_different_task(dataset):
    first, second = load_records(dataset, "train")[:2]
    client = RemoteClient("http://127.0.0.1:9", TOKEN)
    with pytest.raises(ValueError):
        Episode(first.task, sandbox=RemoteSandbox(second.task, client))
    with pytest.raises(ValueError):
        Episode(first.task, image="other:image", sandbox=RemoteSandbox(first.task, client))


def test_remote_sandbox_cannot_run_private_checks(dataset):
    record = load_records(dataset, "train")[0]
    sandbox = RemoteSandbox(record.task, RemoteClient("http://127.0.0.1:9", TOKEN))
    with pytest.raises(SandboxError):
        sandbox.run_check("solution", "solve", [[]], {})


def test_token_file_is_created_private_and_reused(tmp_path):
    path = tmp_path / "secret" / "token"
    token = ensure_token(path)
    assert len(token) >= 32
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert ensure_token(path) == token
    path.chmod(0o644)
    with pytest.raises(ValueError, match="permissions"):
        ensure_token(path)


sandbox_only = pytest.mark.skipif(
    os.environ.get("TOOLSMITH_RUN_SANDBOX_TESTS") != "1",
    reason="Set TOOLSMITH_RUN_SANDBOX_TESTS=1 after building the sandbox image",
)


@pytest.mark.sandbox
@sandbox_only
def test_remote_reference_episode_matches_local_outcome_and_hides_private_data(server, dataset):
    service, url = server
    client = RemoteClient(url, TOKEN)
    record = load_records(dataset, "dev")[0]
    fixed = apply_patch(record.task, record.reference_patch).source_files[0].content
    with Episode(record.task, sandbox=RemoteSandbox(record.task, client)) as episode:
        episode.dispatch("read_file", {"path": SOURCE_PATH})
        failed = episode.dispatch("run_visible_tests", {})
        episode.dispatch("edit_file", {"path": SOURCE_PATH, "content": fixed})
        passed = episode.dispatch("run_visible_tests", {})
        episode.dispatch("submit_patch", {})
    assert failed.observation["status"] == "failed"
    assert passed.observation["status"] == "passed"
    assert episode.image_id.startswith("sha256:")

    score = client.score(record.task.task_id, episode.submitted_patch)
    assert score["status"] == "passed" and score["reward"] == 1
    assert set(score) == {"status", "success", "reward", "patch_sha256", "image_id"}
    response_text = json.dumps(score)
    for case in record.checks.cases:
        assert json.dumps(case.expected) not in response_text or len(json.dumps(case.expected)) < 4

    broken = client.score(record.task.task_id, "not a patch\n")
    assert broken["status"] == "invalid_patch" and broken["reward"] == -1

    logged = [json.loads(line) for line in (service.score_log).read_text().splitlines()]
    assert [row["task_id"] for row in logged] == [record.task.task_id] * 2
    assert len(logged[0]["result"]["cases"]) == len(record.checks.cases)
