import json

import pytest

from toolsmith.reporting import compare, load_run, merge_runs, paired_bootstrap, render_markdown

GENERATION = {"max_new_tokens": 256, "max_context_tokens": 4096, "temperature": 1.0, "top_p": 0.95}
TASKS = ["dev-0000-a", "dev-0001-a", "test_iid-0000-b", "test_iid-0001-b"]


def row(task_id, success, *, terminal="submitted", reward=None, tokens=100, seconds=10.0):
    if reward is None:
        reward = 1 if success else 0
    return {
        "task_id": task_id,
        "success": success,
        "reward": reward,
        "terminal_reason": terminal,
        "new_tokens": tokens,
        "seconds": seconds,
        "tool_calls": 5,
    }


def write_run(tmp_path, name, rows, **config_overrides):
    directory = tmp_path / name
    directory.mkdir()
    config = {
        "model": "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
        "model_revision": "f6c832e6",
        "dataset_manifest_sha256": "016b3bdd",
        "generation": GENERATION,
        "tasks_requested": len(rows),
        "tasks_completed": len(rows),
        "adapter": None,
    } | config_overrides
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return directory


def test_bootstrap_is_seeded_and_brackets_the_paired_difference():
    before = [False] * 30 + [True] * 70
    after = [True] * 85 + [False] * 15
    first = paired_bootstrap(before, after, seed=17, resamples=2000)
    assert first == paired_bootstrap(before, after, seed=17, resamples=2000)
    assert first["difference"] == pytest.approx(0.15)
    assert first["low"] < first["difference"] < first["high"]
    certain = paired_bootstrap([False] * 10, [True] * 10, seed=17, resamples=200)
    assert (certain["low"], certain["high"]) == (1.0, 1.0)


def test_bootstrap_rejects_unpaired_or_empty_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap([True], [True, False])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


def test_comparison_keeps_failures_in_denominators_and_pairs_by_task(tmp_path):
    base = write_run(
        tmp_path,
        "base",
        [
            row("dev-0000-a", True),
            row("dev-0001-a", False, terminal="truncated"),
            row("test_iid-0000-b", True),
            row("test_iid-0001-b", False, terminal="submitted"),  # wrong patch
        ],
    )
    tuned = write_run(tmp_path, "tuned", [row(t, True, tokens=50) for t in TASKS])
    report = compare([load_run(base, "base"), load_run(tuned, "tuned")], resamples=200)
    by_name = {run["name"]: run for run in report["runs"]}
    assert by_name["base"]["overall"]["successes"] == 2
    assert by_name["base"]["overall"]["episodes"] == 4
    assert by_name["base"]["overall"]["wrong_submissions"] == 1
    assert by_name["base"]["splits"]["dev"]["terminal_reasons"] == {"submitted": 1, "truncated": 1}
    assert by_name["tuned"]["overall"]["new_tokens_per_success"] == 50
    [pair] = report["pairs"]
    assert (pair["reference"], pair["candidate"]) == ("base", "tuned")
    assert pair["overall"]["gained"] == 2 and pair["overall"]["lost"] == 0
    assert pair["overall"]["difference"] == pytest.approx(0.5)
    assert "| tuned |" in render_markdown(report)


def test_zero_successes_leave_per_success_costs_undefined(tmp_path):
    none = write_run(tmp_path, "none", [row(t, False, terminal="truncated") for t in TASKS])
    report = compare([load_run(none, "none")], resamples=10)
    overall = report["runs"][0]["overall"]
    assert overall["success_rate"] == 0
    assert overall["new_tokens_per_success"] is None
    assert overall["seconds_per_success"] is None


def test_infrastructure_errors_are_reported_not_dropped(tmp_path):
    rows = [row(t, True) for t in TASKS[:3]] + [
        row(TASKS[3], False, terminal="infrastructure_error", reward=None)
    ]
    rows[3]["reward"] = None
    run = write_run(tmp_path, "infra", rows)
    overall = compare([load_run(run, "infra")], resamples=10)["runs"][0]["overall"]
    assert overall["episodes"] == 4 and overall["infrastructure_errors"] == 1


@pytest.mark.parametrize(
    "override",
    [
        {"dataset_manifest_sha256": "different"},
        {"model_revision": "other"},
        {"generation": GENERATION | {"temperature": 0.7}},
    ],
)
def test_runs_with_different_settings_are_not_comparable(tmp_path, override):
    first = write_run(tmp_path, "first", [row(t, True) for t in TASKS])
    second = write_run(tmp_path, "second", [row(t, True) for t in TASKS], **override)
    with pytest.raises(ValueError, match="not comparable"):
        compare([load_run(first, "first"), load_run(second, "second")], resamples=10)


def test_missing_episodes_are_errors_not_smaller_denominators(tmp_path):
    full = write_run(tmp_path, "full", [row(t, True) for t in TASKS])
    partial = write_run(tmp_path, "partial", [row(t, True) for t in TASKS[:3]])
    with pytest.raises(ValueError, match="task sets differ"):
        compare([load_run(full, "full"), load_run(partial, "partial")], resamples=10)
    incomplete = write_run(tmp_path, "incomplete", [row(t, True) for t in TASKS], tasks_requested=5)
    with pytest.raises(ValueError, match="incomplete"):
        load_run(incomplete, "incomplete")


def test_duplicate_task_ids_are_rejected(tmp_path):
    duplicated = write_run(tmp_path, "dup", [row(TASKS[0], True), row(TASKS[0], False)])
    with pytest.raises(ValueError, match="Duplicate"):
        load_run(duplicated, "dup")


def test_shards_merge_into_one_run_that_compares_against_a_full_run(tmp_path):
    full = write_run(tmp_path, "base", [row(t, False) for t in TASKS])
    adapter = {"source": "dataset:a", "sha256": "abc"}
    even = write_run(
        tmp_path, "s0", [row(t, True) for t in TASKS[0::2]], shard=[0, 2], adapter=adapter
    )
    odd = write_run(
        tmp_path, "s1", [row(t, True) for t in TASKS[1::2]], shard=[1, 2], adapter=adapter
    )
    config, rows = merge_runs([even, odd], "tuned")
    assert config["tasks_requested"] == config["tasks_completed"] == len(TASKS)
    assert [s["shard"] for s in config["shards"]] == [[0, 2], [1, 2]]
    assert sorted(r["task_id"] for r in rows) == sorted(TASKS)
    merged = tmp_path / "tuned"
    merged.mkdir()
    (merged / "config.json").write_text(json.dumps(config))
    (merged / "trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    [pair] = compare([load_run(full, "base"), load_run(merged, "tuned")], resamples=10)["pairs"]
    assert pair["overall"]["gained"] == len(TASKS)


def test_shards_with_different_adapters_overlap_or_gaps_are_rejected(tmp_path):
    a = {"source": "dataset:a", "sha256": "abc"}
    s0 = write_run(tmp_path, "s0", [row(t, True) for t in TASKS[0::2]], shard=[0, 2], adapter=a)
    s1 = write_run(
        tmp_path,
        "s1",
        [row(t, True) for t in TASKS[1::2]],
        shard=[1, 2],
        adapter=a | {"sha256": "x"},
    )
    with pytest.raises(ValueError, match="adapter"):
        merge_runs([s0, s1], "m")
    dup = write_run(tmp_path, "dup", [row(t, True) for t in TASKS[0::2]], shard=[1, 2], adapter=a)
    with pytest.raises(ValueError, match="overlap"):
        merge_runs([s0, dup], "m")
    third = write_run(tmp_path, "s0of3", [row(TASKS[0], True)], shard=[0, 3], adapter=a)
    with pytest.raises(ValueError, match="shard"):
        merge_runs([third], "m")
