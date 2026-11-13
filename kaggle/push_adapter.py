"""Upload a trained LoRA adapter as a private dataset that evaluation notebooks mount.

- Copies `adapter_config.json` and `adapter_model.safetensors` from --adapter-dir.
- Adds the base model's license (OpenMDW-1.1) and its notice of origin.
- Creates the dataset, or adds a new version if it exists; prints the weights' SHA-256.

Usage: python kaggle/push_adapter.py --kaggle-user <user> \
           --adapter-dir runs/kaggle-grpo/grpo-base/adapter --slug toolsmith-grpo-base-adapter
"""

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

from upload_dataset import upload

LICENSE = Path(__file__).resolve().parent.parent / "licenses" / "OpenMDW-1.1.txt"
NOTICE = "NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16, by NVIDIA, under OpenMDW-1.1.\n"
FILES = ("adapter_config.json", "adapter_model.safetensors")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-user", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--slug", required=True, help="Dataset slug, e.g. toolsmith-grpo-base-adapter."
    )
    args = parser.parse_args()
    if missing := [name for name in FILES if not (args.adapter_dir / name).is_file()]:
        raise SystemExit(f"{args.adapter_dir} is missing {missing}")
    dataset_id = f"{args.kaggle_user}/{args.slug}"
    staging = Path(tempfile.mkdtemp(prefix="toolsmith-adapter-"))
    try:
        for name in FILES:
            shutil.copyfile(args.adapter_dir / name, staging / name)
        shutil.copyfile(LICENSE, staging / "LICENSE.OpenMDW-1.1.txt")
        (staging / "NOTICE.txt").write_text(NOTICE)
        digest = hashlib.sha256((staging / FILES[1]).read_bytes()).hexdigest()
        action = upload(staging, dataset_id, "Adapter upload")
        print(f"{action} private dataset {dataset_id} (adapter sha256 {digest})")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
