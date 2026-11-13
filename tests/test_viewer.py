import json

import pytest

from toolsmith.dataset import DatasetConfig, generate_dataset, load_records
from toolsmith.reporting import load_run
from toolsmith.routing import RoutingPolicy
from toolsmith.viewer import build_site

SMALL = DatasetConfig(train=3, dev=3, test_iid=3, test_compositional=2, challenge=4)
GENERATION = {"max_new_tokens": 256, "max_context_tokens": 4096, "temperature": 1.0, "top_p": 0.95}


@pytest.fixture
def tasks(tmp_path):
    directory = tmp_path / "data"
    generate_dataset(SMALL, directory)
    return {r.task.task_id: r.task for r in load_records(directory, "dev")}


def write_run(tmp_path, name, tasks, text):
    rows = []
    for task_id in sorted(tasks):
        rows.append(
            {
                "task_id": task_id,
                "success": False,
                "reward": 0,
                "terminal_reason": "malformed_tool_call",
                "new_tokens": 12,
                "seconds": 1.5,
                "tool_calls": 0,
                "events": [],
                "turns": [{"turn": 0, "text": text, "new_tokens": 12, "truncated": False}],
                "submitted_patch": None,
                "score": None,
            }
        )
    directory = tmp_path / name
    directory.mkdir()
    config = {
        "model": "m",
        "model_revision": "r",
        "dataset_manifest_sha256": "d",
        "generation": GENERATION,
        "tasks_requested": len(rows),
        "tasks_completed": len(rows),
    }
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return load_run(directory, name)


def test_model_output_is_escaped_and_every_task_has_a_linked_page(tmp_path, tasks):
    run = write_run(tmp_path, "base", tasks, '<script>alert("x")</script>')
    index = build_site([run], tasks, RoutingPolicy(), tmp_path / "site")
    listing = index.read_text()
    for task_id in tasks:
        assert f'href="tasks/{task_id}.html"' in listing
        page = (tmp_path / "site" / "tasks" / f"{task_id}.html").read_text()
        assert "<script>" not in page
        assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in page
        assert tasks[task_id].issue.split("\n")[0][:40] in page


def test_unconfigured_escalation_is_shown_as_not_run(tmp_path, tasks):
    run = write_run(tmp_path, "base", tasks, "no call")
    build_site([run], tasks, RoutingPolicy(), tmp_path / "site")
    page = (tmp_path / "site" / "tasks" / f"{sorted(tasks)[0]}.html").read_text()
    assert "Upfront decision: <code>small</code>" in page
    assert "not run (no escalation model configured)" in page


def test_runs_without_public_inputs_or_existing_output_are_errors(tmp_path, tasks):
    run = write_run(tmp_path, "base", tasks, "x")
    partial = dict(list(tasks.items())[:1])
    with pytest.raises(ValueError, match="Public inputs missing"):
        build_site([run], partial, RoutingPolicy(), tmp_path / "site-a")
    (tmp_path / "site-b").mkdir()
    with pytest.raises(FileExistsError):
        build_site([run], tasks, RoutingPolicy(), tmp_path / "site-b")
