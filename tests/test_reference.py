import json
import os
from dataclasses import replace

import pytest

from toolsmith.dataset import DatasetConfig, generate_dataset, load_records
from toolsmith.reference import (
    build_demonstration,
    build_demonstrations,
    verify_dataset,
    verify_record,
)
from toolsmith.sandbox import DEFAULT_IMAGE, SandboxLimits
from toolsmith.tasks import BugTask

SMALL = DatasetConfig(train=3, dev=3, test_iid=3, test_compositional=2, challenge=4)
LIMITS = SandboxLimits()


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path / "data"
    generate_dataset(SMALL, directory)
    return directory


def test_demonstrations_refuse_held_out_splits(dataset):
    record = load_records(dataset, "test_iid")[0]
    with pytest.raises(ValueError, match="train and dev"):
        build_demonstration(record, image=DEFAULT_IMAGE, limits=LIMITS)


@pytest.mark.sandbox
@pytest.mark.skipif(
    os.environ.get("TOOLSMITH_RUN_SANDBOX_TESTS") != "1", reason="requires sandbox image"
)
def test_every_split_original_fails_and_reference_passes(dataset, tmp_path):
    report = verify_dataset(dataset, tmp_path / "verification", workers=4)
    assert report["success"], report["failures"]
    assert report["verified"] == report["total"] == sum(report["counts"].values())
    assert report["image_id"].startswith("sha256:")
    rows = [
        json.loads(line)
        for line in (tmp_path / "verification/records.jsonl").read_text().splitlines()
    ]
    assert all(
        row["original_fails_visibly"] and row["private_passed"] == row["private_total"]
        for row in rows
    )
    with pytest.raises(FileExistsError):
        verify_dataset(dataset, tmp_path / "verification")


@pytest.mark.sandbox
@pytest.mark.skipif(
    os.environ.get("TOOLSMITH_RUN_SANDBOX_TESTS") != "1", reason="requires sandbox image"
)
def test_fixture_whose_original_already_passes_is_rejected(dataset):
    record = load_records(dataset, "train")[0]
    fixed_source = record.task.source_files[0].content
    # A "bug" whose visible test passes on the original is a broken fixture.
    healthy = BugTask.from_dict(
        {
            "schema_version": 1,
            "task_id": record.task.task_id,
            "issue": record.task.issue,
            "source_files": {"src/solution.py": fixed_source},
            "visible_tests": {
                "tests/test_solution.py": "from solution import solve\n\n"
                "def test_runs():\n    assert solve([]) == []\n"
            },
        }
    )
    result = verify_record(replace(record, task=healthy), image=DEFAULT_IMAGE, limits=LIMITS)
    assert result["original_fails_visibly"] is False
    assert result["success"] is False


@pytest.mark.sandbox
@pytest.mark.skipif(
    os.environ.get("TOOLSMITH_RUN_SANDBOX_TESTS") != "1", reason="requires sandbox image"
)
def test_wrong_reference_patch_fails_private_checks(dataset):
    records = load_records(dataset, "train")
    # Another task's patch cannot apply to this task's distinct source.
    result = verify_record(
        replace(records[0], reference_patch=records[1].reference_patch),
        image=DEFAULT_IMAGE,
        limits=LIMITS,
    )
    assert result["success"] is False
    assert result["reference_status"] == "invalid_patch"


@pytest.mark.sandbox
@pytest.mark.skipif(
    os.environ.get("TOOLSMITH_RUN_SANDBOX_TESTS") != "1", reason="requires sandbox image"
)
def test_demonstrations_replay_and_contain_no_private_data(dataset, tmp_path):
    output = tmp_path / "demos"
    manifest = build_demonstrations(dataset, output, workers=4)
    assert manifest["success"], manifest["failures"]
    assert manifest["counts"] == {"train": SMALL.train, "dev": SMALL.dev}
    private_values = set()
    for split in ("train", "dev"):
        for record in load_records(dataset, split):
            for case in record.checks.cases:
                private_values.add(json.dumps(case.args))
    for split in ("train", "dev"):
        for line in (output / f"{split}.jsonl").read_text().splitlines():
            transcript = json.loads(line)
            assert [e["tool"] for e in transcript["events"]] == [
                "read_file",
                "run_visible_tests",
                "edit_file",
                "run_visible_tests",
                "submit_patch",
            ]
            assert transcript["terminal_reason"] == "submitted"
            assert not {"checks", "reference_patch", "metadata"} & set(transcript)
            assert not {"checks", "expected", "reference_patch"} & set(
                transcript["initial_context"]
            )
            serialized = json.dumps(transcript["initial_context"])
            assert not any(value in serialized for value in private_values if len(value) > 4)
    verification = [
        json.loads(line)
        for line in (output / "private/verification.jsonl").read_text().splitlines()
    ]
    assert all(
        row["replay_matches"] and row["evaluation_status"] == "passed" for row in verification
    )
