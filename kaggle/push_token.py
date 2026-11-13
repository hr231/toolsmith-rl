"""Upload the sandbox bearer token as a private dataset that GPU notebooks mount.

- Reads the token file (enforcing 0600 permissions); never prints the token.
- Writes it to a temporary 0700 staging directory as `sandbox_token.txt`.
- Creates the dataset, or adds a new version if it exists, then deletes the staging directory.
- Rerun after `toolsmith sandbox-serve --rotate-token` so only the current token is valid.

Usage: python kaggle/push_token.py --kaggle-user <user>
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from upload_dataset import upload

from toolsmith.remote import ensure_token

SLUG = "toolsmith-sandbox-token"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-user", required=True)
    parser.add_argument(
        "--token-file", type=Path, default=Path.home() / ".toolsmith" / "sandbox_token"
    )
    args = parser.parse_args()
    if not args.token_file.exists():
        raise SystemExit(f"No token at {args.token_file}; start sandbox-serve first")
    token = ensure_token(args.token_file)  # also enforces 0600 permissions
    dataset_id = f"{args.kaggle_user}/{SLUG}"
    staging = Path(tempfile.mkdtemp(prefix="toolsmith-token-"))
    try:
        staging.chmod(0o700)
        secret = staging / "sandbox_token.txt"
        secret.write_text(token + "\n")
        secret.chmod(0o600)
        action = upload(staging, dataset_id, "Rotated sandbox token")
        print(f"{action} private dataset {dataset_id}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
