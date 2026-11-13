"""Create or version a private dataset from a staged directory.

- Writes `dataset-metadata.json` for the dataset ID into the staging directory.
- Adds a new version when the dataset exists; otherwise creates it (private by default).
- Raises SystemExit with the CLI's error text when the upload fails.
"""

import json
import subprocess
from pathlib import Path


def upload(staging: Path, dataset_id: str, message: str) -> str:
    """Upload every file in `staging` as `dataset_id`; return "Created" or "Updated"."""
    (staging / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": dataset_id.split("/", 1)[1],
                "id": dataset_id,
                "licenses": [{"name": "other"}],
            }
        )
    )
    status = ["kaggle", "datasets", "status", dataset_id]
    exists = subprocess.run(status, capture_output=True, text=True).returncode == 0
    command = (
        ["kaggle", "datasets", "version", "-p", str(staging), "-m", message]
        if exists
        else ["kaggle", "datasets", "create", "-p", str(staging)]
    )
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Dataset upload failed: {result.stderr[-500:]}")
    return "Updated" if exists else "Created"
