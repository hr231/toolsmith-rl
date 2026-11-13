"""Deterministic public/private repair datasets with regeneration-based validation.

- Generates split datasets from a seed and per-split counts (CPU-only).
- Fingerprints programs via `ast` parsing; generated sources are never executed.
- Validates datasets by regenerating them and comparing bytes.
- Execution checks of failures and reference repairs live in `toolsmith.reference`.
"""

import ast
import copy
import difflib
import hashlib
import json
import os
import shutil
import stat
import tempfile
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from toolsmith.evaluator import _MISSING, CheckSpec, PrivateChecks, apply_patch
from toolsmith.tasks import BugTask
from toolsmith.templates import (
    CHALLENGE_CASES,
    FAMILY_ASSIGNMENTS,
    IID_CAVEAT,
    MAX_SPLIT_COUNT,
    PROVENANCE,
    make_template,
)

SPLITS = tuple(FAMILY_ASSIGNMENTS)
SCHEMA_VERSION = 1
GENERATOR_VERSION = "toolsmith-repair-v1"
FINGERPRINT_VERSION = "ast-alpha-pair-v1"
SOURCE_PATH = "src/solution.py"
TEST_PATH = "tests/test_solution.py"
DEMOS_DIRECTORY = "demos"


@dataclass(frozen=True)
class DatasetConfig:
    seed: int = 17
    train: int = 500
    dev: int = 100
    test_iid: int = 100
    test_compositional: int = 100
    challenge: int = 40

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= 10_000:
            raise ValueError("seed must be an integer in [0, 10000]")
        for split in SPLITS:
            value = getattr(self, split)
            limit = len(CHALLENGE_CASES) if split == "challenge" else MAX_SPLIT_COUNT
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"{split} must be an integer in [1, {limit}]")


@dataclass(frozen=True)
class DatasetRecord:
    task: BugTask
    checks: PrivateChecks
    reference_patch: str
    metadata: dict


def load_config(path: Path) -> DatasetConfig:
    """Read exactly [dataset] seed plus every split count; unknown keys are errors."""
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    names = {field.name for field in fields(DatasetConfig)}
    if set(data) != {"dataset"} or type(data["dataset"]) is not dict:
        raise ValueError("Dataset config must contain exactly one [dataset] table")
    if set(data["dataset"]) != names:
        raise ValueError(f"[dataset] requires exactly: {', '.join(sorted(names))}")
    return DatasetConfig(**data["dataset"])


def task_manifest(task: BugTask) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task.task_id,
        "issue": task.issue,
        "source_files": {file.path: file.content for file in task.source_files},
        "visible_tests": {file.path: file.content for file in task.visible_tests},
    }


def checks_manifest(checks: PrivateChecks) -> dict:
    cases = []
    for case in checks.cases:
        row = {
            "id": case.id,
            "module": case.module,
            "function": case.function,
            "args": copy.deepcopy(case.args),
            "kwargs": copy.deepcopy(case.kwargs),
        }
        if case.raises is not None:
            row["raises"] = case.raises
        else:
            row["expected"] = copy.deepcopy(case.expected)
        cases.append(row)
    return {"schema_version": SCHEMA_VERSION, "task_id": checks.task_id, "cases": cases}


class _Normalizer(ast.NodeTransformer):
    """Drop docstrings and alpha-rename bound names with one module-wide map."""

    def __init__(self, names: dict[str, str]) -> None:
        self._names = names

    def _strip_docstring(self, node):
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                node.body = body[1:]
        return node

    def visit_Module(self, node):
        return self.generic_visit(self._strip_docstring(node))

    def visit_FunctionDef(self, node):
        node.name = self._names.get(node.name, node.name)
        return self.generic_visit(self._strip_docstring(node))

    def visit_arg(self, node):
        node.arg = self._names.get(node.arg, node.arg)
        node.annotation = None
        return node

    def visit_Name(self, node):
        node.id = self._names.get(node.id, node.id)
        return node


def _normalized_dump(source: str) -> str:
    tree = ast.parse(source)
    bound = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            bound.append(node.name)
        elif isinstance(node, ast.arg):
            bound.append(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.append(node.id)
    names = {}
    for name in bound:
        names.setdefault(name, f"_v{len(names)}")
    # Builtins and attributes (min/max/len/split) keep their meaning-bearing names.
    return ast.dump(_Normalizer(names).visit(tree), annotate_fields=False)


def source_fingerprint(task: BugTask, fixed: BugTask) -> str:
    """Hash the normalized (buggy, fixed) program pair; tests, issue, and IDs excluded."""
    pair = [
        [_normalized_dump(file.content) for file in task.source_files],
        [_normalized_dump(file.content) for file in fixed.source_files],
    ]
    payload = json.dumps([FINGERPRINT_VERSION, pair], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _visible_test(value: object, expected: object) -> str:
    return (
        "from solution import solve\n\n\n"
        "def test_reported_behavior():\n"
        f"    assert solve({value!r}) == {expected!r}\n"
    )


def _patch(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{SOURCE_PATH}",
            tofile=f"b/{SOURCE_PATH}",
        )
    )


def _record(split: str, seed: int, index: int) -> DatasetRecord:
    instance = make_template(split, seed, index)
    task_id = f"{split}-{index:04d}-{instance.family}"
    task = BugTask.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "issue": instance.issue,
            "source_files": {SOURCE_PATH: instance.source},
            "visible_tests": {
                TEST_PATH: _visible_test(instance.visible_input, instance.visible_expected)
            },
        }
    )
    cases = tuple(
        CheckSpec(f"case-{number}", "solution", "solve", [copy.deepcopy(value)], {}, expected=out)
        for number, (value, out) in enumerate(instance.private_cases)
    )
    if not 2 <= len(cases) <= 4:
        raise ValueError(f"{task_id}: templates must produce two to four private cases")
    reference_patch = _patch(instance.source, instance.fixed_source)
    fixed = apply_patch(task, reference_patch)
    if fixed.source_files[0].content != instance.fixed_source:
        raise ValueError(f"{task_id}: reference patch does not reconstruct the fixed source")
    metadata = {
        "split": split,
        "project_family": f"{instance.family}_project",
        "template_family": instance.family,
        "bug_family": instance.bug_family,
        "composition_family": instance.composition_family,
        "seed": seed,
        "instance_index": index,
        "license": "MIT",
        "provenance": PROVENANCE,
        "source_fingerprint": source_fingerprint(task, fixed),
    }
    if split == "challenge":
        metadata["review_note"] = instance.review_note
    return DatasetRecord(task, PrivateChecks(task_id, cases), reference_patch, metadata)


def _build(config: DatasetConfig) -> dict[str, tuple[DatasetRecord, ...]]:
    splits = {
        split: tuple(_record(split, config.seed, index) for index in range(getattr(config, split)))
        for split in SPLITS
    }
    seen_ids, seen_fingerprints, family_owner = set(), set(), {}
    for split, records in splits.items():
        for record in records:
            if record.task.task_id in seen_ids:
                raise ValueError(f"Duplicate task id {record.task.task_id}")
            seen_ids.add(record.task.task_id)
            fingerprint = record.metadata["source_fingerprint"]
            if fingerprint in seen_fingerprints:
                raise ValueError(f"Duplicate normalized program pair: {record.task.task_id}")
            seen_fingerprints.add(fingerprint)
            for key in ("project_family", "template_family"):
                owner = family_owner.setdefault((key, record.metadata[key]), split)
                if owner != split:
                    raise ValueError(f"{record.metadata[key]} appears in {owner} and {split}")
            if (record.metadata["composition_family"] is not None) != (
                split == "test_compositional"
            ):
                raise ValueError(f"{record.task.task_id}: compositions are held out")
    return splits


def _line(value: object) -> str:
    return (
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    )


def _render(config: DatasetConfig) -> dict[str, bytes]:
    files = {}
    for split, records in _build(config).items():
        files[f"{split}.jsonl"] = "".join(_line(task_manifest(r.task)) for r in records).encode()
        files[f"private/{split}.jsonl"] = "".join(
            _line(
                {
                    "task_id": r.task.task_id,
                    "checks": checks_manifest(r.checks),
                    "reference_patch": r.reference_patch,
                    "metadata": r.metadata,
                }
            )
            for r in records
        ).encode()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "fingerprint_version": FINGERPRINT_VERSION,
        "config": {field.name: getattr(config, field.name) for field in fields(config)},
        "family_assignments": {split: list(names) for split, names in FAMILY_ASSIGNMENTS.items()},
        "counts": {split: getattr(config, split) for split in SPLITS},
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
        "license": "MIT",
        "provenance": PROVENANCE,
        "iid_caveat": IID_CAVEAT,
        "execution_verified": False,
    }
    files["manifest.json"] = _line(manifest).encode()
    return files


def _summary(files: dict[str, bytes]) -> dict:
    manifest = json.loads(files["manifest.json"])
    return {
        "counts": manifest["counts"],
        "files": manifest["files"],
        "manifest_sha256": hashlib.sha256(files["manifest.json"]).hexdigest(),
    }


def generate_dataset(config: DatasetConfig, output: Path) -> dict:
    """Write a new dataset directory atomically; existing destinations are refused."""
    if not isinstance(config, DatasetConfig):
        raise ValueError("config must be DatasetConfig")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Dataset destination already exists: {output}")
    files = _render(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (staging / "private").mkdir()
        for name, data in files.items():
            with (staging / name).open("xb") as stream:
                stream.write(data)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _summary(files)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _regular_bytes(path: Path) -> bytes:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Dataset file is not a regular file: {path}")
    return path.read_bytes()


def _directory(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Dataset directory must be a real directory: {directory}")


def _manifest_config(directory: Path) -> DatasetConfig:
    raw = _regular_bytes(directory / "manifest.json")
    manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    if type(manifest) is not dict or manifest.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("Unsupported or malformed dataset manifest")
    config = manifest.get("config")
    if type(config) is not dict:
        raise ValueError("Dataset manifest lacks a config object")
    return DatasetConfig(**config)


def validate_dataset(directory: Path) -> dict:
    """Regenerate from the recorded config and require byte-identical files.

    Any relabelled family, poisoned expected value, altered source, duplicate ID,
    or stale fingerprint is rejected even if an attacker recomputed the hashes.
    Only an optional `demos/` output directory is tolerated and never read.
    """
    _directory(directory)
    expected = _render(_manifest_config(directory))
    allowed_root = {name for name in expected if "/" not in name} | {"private", DEMOS_DIRECTORY}
    present_root = {entry.name for entry in directory.iterdir()}
    if present_root - allowed_root:
        raise ValueError(f"Unexpected dataset entries: {sorted(present_root - allowed_root)}")
    _directory(directory / "private")
    private_expected = {name.split("/", 1)[1] for name in expected if name.startswith("private/")}
    present_private = {entry.name for entry in (directory / "private").iterdir()}
    if present_private != private_expected:
        raise ValueError("Private dataset files do not match the manifest")
    for name, data in expected.items():
        if _regular_bytes(directory / name) != data:
            raise ValueError(f"{name} differs from deterministic regeneration")
    return _summary(expected)


def load_records(directory: Path, split: str) -> tuple[DatasetRecord, ...]:
    """Load one split after verifying both of its files regenerate byte-identically."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {', '.join(SPLITS)}")
    _directory(directory)
    config = _manifest_config(directory)
    records = tuple(_record(split, config.seed, i) for i in range(getattr(config, split)))
    public = "".join(_line(task_manifest(r.task)) for r in records).encode()
    private = "".join(
        _line(
            {
                "task_id": r.task.task_id,
                "checks": checks_manifest(r.checks),
                "reference_patch": r.reference_patch,
                "metadata": r.metadata,
            }
        )
        for r in records
    ).encode()
    if _regular_bytes(directory / f"{split}.jsonl") != public:
        raise ValueError(f"{split}.jsonl differs from deterministic regeneration")
    if _regular_bytes(directory / "private" / f"{split}.jsonl") != private:
        raise ValueError(f"private/{split}.jsonl differs from deterministic regeneration")
    # Return parsed on-disk rows (now proven identical) as fresh, unshared objects.
    loaded = []
    for public_line, private_line in zip(
        public.decode().splitlines(), private.decode().splitlines(), strict=True
    ):
        task = BugTask.from_dict(json.loads(public_line, object_pairs_hook=_unique_object))
        row = json.loads(private_line, object_pairs_hook=_unique_object)
        cases = tuple(
            CheckSpec(
                case["id"],
                case["module"],
                case["function"],
                case["args"],
                case["kwargs"],
                expected=case.get("expected", _MISSING),
                raises=case.get("raises"),
            )
            for case in row["checks"]["cases"]
        )
        loaded.append(
            DatasetRecord(
                task, PrivateChecks(row["task_id"], cases), row["reference_patch"], row["metadata"]
            )
        )
    return tuple(loaded)
