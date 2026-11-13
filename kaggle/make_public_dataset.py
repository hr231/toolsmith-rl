"""Stage the public part of a validated dataset and the Toolsmith wheel for upload.

- Validates the dataset, then copies the public split files and the manifest.
- Copies the train/dev demonstrations as top-level `demos_<name>` files.
- Never copies `private/` (checks, reference patches, metadata); asserts none was staged.
- Writes `dataset-metadata.json` and prints a short digest of every staged file.

Usage: python kaggle/make_public_dataset.py --directory data/generated/repair-v1 \
           --wheel dist/toolsmith_rl-0.1.0-py3-none-any.whl --output build/kaggle-public \
           --dataset-id <user>/toolsmith-repair-public
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from toolsmith.dataset import SPLITS, validate_dataset

# Demonstration files copied as top-level files because the uploader skips subdirectories.
PUBLIC_DEMOS = ("train.jsonl", "dev.jsonl", "manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()
    summary = validate_dataset(args.directory)
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    files = [f"{split}.jsonl" for split in SPLITS] + ["manifest.json"]
    for name in files:
        shutil.copyfile(args.directory / name, args.output / name)
    for name in PUBLIC_DEMOS:
        shutil.copyfile(args.directory / "demos" / name, args.output / f"demos_{name}")
    shutil.copyfile(args.wheel, args.output / args.wheel.name)
    staged = sorted(path for path in args.output.rglob("*") if path.is_file())
    assert not any("private" in path.relative_to(args.output).parts for path in staged)
    (args.output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": args.dataset_id.split("/", 1)[1],
                "id": args.dataset_id,
                "licenses": [{"name": "other"}],
                "description": "Public task manifests, verified demonstrations, and the Toolsmith "
                f"wheel. Dataset manifest SHA-256 {summary['manifest_sha256']}. No private checks.",
            },
            indent=2,
        )
        + "\n"
    )
    for path in staged:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f"{digest}  {path.relative_to(args.output)}")


if __name__ == "__main__":
    main()
