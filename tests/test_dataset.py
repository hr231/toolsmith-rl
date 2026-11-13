import hashlib
import json
from dataclasses import replace

import pytest

from toolsmith.dataset import (
    DatasetConfig,
    checks_manifest,
    generate_dataset,
    load_config,
    load_records,
    source_fingerprint,
    task_manifest,
    validate_dataset,
)
from toolsmith.evaluator import CheckSpec, PrivateChecks, apply_patch
from toolsmith.templates import CHALLENGE_CASES, FAMILY_ASSIGNMENTS

SMALL = DatasetConfig(train=6, dev=3, test_iid=3, test_compositional=2, challenge=4)


def rewrite_row(directory, relative, change):
    path = directory / relative
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows[0])
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            for row in rows
        )
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    )


def test_defaults_and_strict_toml(tmp_path):
    assert sum(getattr(DatasetConfig(), split) for split in FAMILY_ASSIGNMENTS) == 840
    config = tmp_path / "data.toml"
    config.write_text(
        "[dataset]\nseed=17\ntrain=6\ndev=3\ntest_iid=3\ntest_compositional=2\nchallenge=4\n"
    )
    assert load_config(config) == SMALL
    config.write_text(config.read_text() + "unknown=1\n")
    with pytest.raises(ValueError):
        load_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("seed", -1),
        ("train", 0),
        ("dev", True),
        ("test_iid", 1.5),
        ("challenge", 41),
        ("train", 100001),
    ],
)
def test_config_bounds(field, value):
    with pytest.raises(ValueError):
        replace(SMALL, **{field: value})


def test_generation_is_byte_identical_and_loads_roundtrip(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    summary = generate_dataset(SMALL, first)
    assert generate_dataset(SMALL, second) == summary
    for path in first.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()
    assert validate_dataset(first) == summary
    records = load_records(first, "train")
    assert len(records) == SMALL.train
    for record in records:
        assert set(task_manifest(record.task)) == {
            "schema_version",
            "task_id",
            "issue",
            "source_files",
            "visible_tests",
        }
        fixed = apply_patch(record.task, record.reference_patch)
        assert fixed.source_files != record.task.source_files
        assert record.metadata["source_fingerprint"] == source_fingerprint(record.task, fixed)
        assert 2 <= len(record.checks.cases) <= 4
        assert record.metadata["license"] == "MIT"
    with pytest.raises(FileExistsError):
        generate_dataset(SMALL, first)
    with pytest.raises(ValueError):
        load_records(first, "../train")


def test_full_dataset_has_unique_semantic_pairs_and_disjoint_families(tmp_path):
    directory = tmp_path / "full"
    generate_dataset(DatasetConfig(), directory)
    assert validate_dataset(directory)["counts"] == {
        "train": 500,
        "dev": 100,
        "test_iid": 100,
        "test_compositional": 100,
        "challenge": 40,
    }
    fingerprints = set()
    projects, templates = {}, {}
    for split in FAMILY_ASSIGNMENTS:
        for record in load_records(directory, split):
            fingerprint = record.metadata["source_fingerprint"]
            assert fingerprint not in fingerprints
            fingerprints.add(fingerprint)
            for seen, key in [(projects, "project_family"), (templates, "template_family")]:
                name = record.metadata[key]
                assert seen.setdefault(name, split) == split
            assert (record.metadata["composition_family"] is not None) == (
                split == "test_compositional"
            )
    assert len(fingerprints) == 840


@pytest.mark.parametrize(
    "relative,change",
    [
        (
            "private/train.jsonl",
            lambda row: row["metadata"].update(template_family="filter_ceiling"),
        ),
        ("private/train.jsonl", lambda row: row["metadata"].update(project_family="other")),
        ("private/train.jsonl", lambda row: row["metadata"].update(source_fingerprint="0" * 64)),
        ("private/train.jsonl", lambda row: row["checks"]["cases"][0].update(expected="poison")),
        (
            "train.jsonl",
            lambda row: row["source_files"].update(
                {"src/solution.py": "raise RuntimeError('contaminated')\n"}
            ),
        ),
        ("train.jsonl", lambda row: row.update(expected="private leak")),
        ("train.jsonl", lambda row: row.update(task_id="duplicate")),
    ],
)
def test_regeneration_rejects_contamination_even_with_rehashed_files(tmp_path, relative, change):
    generate_dataset(SMALL, tmp_path / "data")
    rewrite_row(tmp_path / "data", relative, change)
    with pytest.raises(ValueError):
        validate_dataset(tmp_path / "data")
    with pytest.raises(ValueError):
        load_records(tmp_path / "data", "train")


def test_hash_schema_and_path_boundaries(tmp_path):
    directory = tmp_path / "data"
    generate_dataset(SMALL, directory)
    path = directory / "dev.jsonl"
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError):
        validate_dataset(directory)
    path.write_bytes(original)
    extra = directory / "unexpected.jsonl"
    extra.write_text("{}\n")
    with pytest.raises(ValueError):
        validate_dataset(directory)
    extra.unlink()
    path.unlink()
    path.symlink_to(directory / "train.jsonl")
    with pytest.raises(ValueError):
        validate_dataset(directory)


def test_duplicate_json_keys_are_not_silently_accepted(tmp_path):
    directory = tmp_path / "data"
    generate_dataset(SMALL, directory)
    path = directory / "manifest.json"
    path.write_text(path.read_text().replace('"seed":17', '"seed":17,"seed":17'))
    with pytest.raises(ValueError):
        validate_dataset(directory)


def test_challenge_specs_are_explicit_and_all_have_review_notes(tmp_path):
    assert len(CHALLENGE_CASES) == 40
    directory = tmp_path / "data"
    generate_dataset(replace(SMALL, challenge=40), directory)
    records = load_records(directory, "challenge")
    assert len({record.metadata["review_note"] for record in records}) == 40
    assert all("human-reviewed" not in record.metadata["review_note"] for record in records)
    assert {record.metadata["template_family"] for record in records} == set(
        FAMILY_ASSIGNMENTS["challenge"]
    )


def test_normalized_pair_fingerprint_ignores_names_comments_and_docstrings():
    from toolsmith.tasks import BugTask

    def task(source):
        return BugTask.from_dict(
            {
                "schema_version": 1,
                "task_id": "ast",
                "issue": "AST",
                "source_files": {"src/solution.py": source},
                "visible_tests": {"tests/test_solution.py": "assert True\n"},
            }
        )

    original = task(
        'def first(x):\n    "doc"\n    return x + 1\ndef second(y):\n    return first(y)\n'
    )
    renamed = task("# comment\ndef alpha(a):\n    return a+1\ndef beta(b):\n    return alpha(b)\n")
    fixed = task("def first(x):\n    return x + 2\ndef second(y):\n    return first(y)\n")
    renamed_fixed = task("def alpha(a):\n    return a+2\ndef beta(b):\n    return alpha(b)\n")
    assert source_fingerprint(original, fixed) == source_fingerprint(renamed, renamed_fixed)
    assert source_fingerprint(original, fixed) != source_fingerprint(original, original)
    different_call = task("def alpha(a):\n    return a+1\ndef beta(b):\n    return beta(b)\n")
    assert source_fingerprint(different_call, renamed_fixed) != source_fingerprint(original, fixed)


def test_checks_serialization_is_defensive_and_preserves_raises():
    checks = PrivateChecks(
        "case",
        (
            CheckSpec("value", "solution", "solve", [[1]], {}, expected=[2]),
            CheckSpec("error", "solution", "solve", [None], {}, raises="TypeError"),
        ),
    )
    manifest = checks_manifest(checks)
    assert "expected" not in manifest["cases"][1]
    assert "raises" not in manifest["cases"][0]
    manifest["cases"][0]["args"][0].append(999)
    manifest["cases"][0]["expected"].append(999)
    assert checks.cases[0].args == [[1]]
    assert checks.cases[0].expected == [2]
