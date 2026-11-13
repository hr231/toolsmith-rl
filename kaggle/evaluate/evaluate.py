"""Evaluate the pinned model, optionally with a LoRA adapter, on T4 GPUs.

- RUN_CONFIG (set by kaggle/push.py): sandbox URL, splits, task limit, run name, and
  an optional adapter source (notebook output or dataset) with its adapter directory.
- Reads the sandbox bearer token from the attached token dataset.
- Loads one model per GPU (adapter merged into the weights) and runs one episode
  per task through the remote sandbox service, one worker thread per GPU.
- Appends every finished episode to partial.jsonl, stops starting new episodes after
  11 hours, then writes config.json, trajectories.jsonl, and summary.json.
"""

import glob
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUN_CONFIG = {}  # replaced by kaggle/push.py
MODEL = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
REVISION = "f6c832e6963ddb74f62c75813a3e8c74b1af67b1"
TRANSFORMERS_COMMIT = "fcc6924dfbac964f00787859b568c867ef5ad4cb"
PINS = [
    f"transformers @ git+https://github.com/shan18/transformers@{TRANSFORMERS_COMMIT}",
    "peft==0.21.2",
    "trl==1.17.1",
]
WORK = "/kaggle/working"
STOP_AFTER_SECONDS = 11 * 3600  # leave time to write results before the 12-hour cap
STARTED = time.time()

# Fail before installing anything if the token is missing.
TOKEN_FILES = glob.glob("/kaggle/input/**/sandbox_token.txt", recursive=True)
if len(TOKEN_FILES) != 1:
    raise SystemExit(f"Expected one sandbox_token.txt in inputs, found {len(TOKEN_FILES)}")
with open(TOKEN_FILES[0]) as stream:
    SANDBOX_TOKEN = stream.read().strip()

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS], check=True)
WHEEL = glob.glob("/kaggle/input/**/toolsmith_rl-*.whl", recursive=True)[0]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--ignore-requires-python", WHEEL],
    check=True,
)
DATA = os.path.dirname(WHEEL)
ADAPTER = None
if RUN_CONFIG.get("adapter_source"):
    # Select exactly one adapter directory among the attached inputs.
    pattern = f"/kaggle/input/**/{RUN_CONFIG['adapter_dir']}/adapter_config.json"
    configs = glob.glob(pattern, recursive=True)
    if len(configs) != 1:
        raise SystemExit(f"Expected one adapter matching {pattern}, found {configs}")
    ADAPTER = os.path.dirname(configs[0])
    with open(os.path.join(ADAPTER, "adapter_model.safetensors"), "rb") as stream:
        ADAPTER_SHA256 = hashlib.sha256(stream.read()).hexdigest()

import importlib.metadata as metadata  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from toolsmith.evaluation import GenerationConfig, HFPolicy, run_episode, write_run  # noqa: E402
from toolsmith.remote import RemoteClient  # noqa: E402
from toolsmith.tasks import BugTask  # noqa: E402

client = RemoteClient(RUN_CONFIG["sandbox_url"], SANDBOX_TOKEN)
print("sandbox health:", client.health(), flush=True)

generation = GenerationConfig(seed=RUN_CONFIG["seed"])
tasks = []
for split in RUN_CONFIG["splits"]:
    with open(os.path.join(DATA, f"{split}.jsonl")) as stream:
        rows = [json.loads(line) for line in stream.read().splitlines()]
    tasks += [BugTask.from_dict(row) for row in rows[: RUN_CONFIG["limit"]]]
# Shard [index, count]: keep every count-th task starting at index.
shard_index, shard_count = RUN_CONFIG.get("shard") or (0, 1)
tasks = tasks[shard_index::shard_count]
with open(os.path.join(DATA, "manifest.json"), "rb") as stream:
    manifest_sha256 = hashlib.sha256(stream.read()).hexdigest()

tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
policies = []
for device in devices:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float16, attn_implementation="sdpa"
    ).to(device)
    if ADAPTER:
        from peft import PeftModel

        # Merge the adapter into the weights for base-model generation speed.
        model = PeftModel.from_pretrained(model, ADAPTER).to(device).merge_and_unload()
    policies.append(HFPolicy(model.eval(), tokenizer, generation, device=device))

partial_path = os.path.join(WORK, "partial.jsonl")
lock = threading.Lock()
rows = []


def work(index_and_task):
    index, task = index_and_task
    if time.time() - STARTED > STOP_AFTER_SECONDS:
        return
    trace = run_episode(task, policies[index % len(policies)], client, generation)
    with lock:
        rows.append(trace)
        with open(partial_path, "a") as stream:
            stream.write(json.dumps(trace, sort_keys=True) + "\n")
        print(
            f"{len(rows)}/{len(tasks)} {trace['task_id']} {trace['terminal_reason']} "
            f"reward={trace['reward']}",
            flush=True,
        )


with ThreadPoolExecutor(max_workers=len(policies)) as pool:
    list(pool.map(work, enumerate(tasks)))

config = {
    **RUN_CONFIG,
    "model": MODEL,
    "model_revision": REVISION,
    "transformers_commit": TRANSFORMERS_COMMIT,
    "versions": {
        name: metadata.version(name)
        for name in ("torch", "transformers", "peft", "trl", "toolsmith-rl")
    },
    "generation": generation.__dict__,
    "dtype": "float16",
    "attention": "sdpa",
    "loops": policies[0]._model.config.n_recurrent_loops,
    "gpus": [torch.cuda.get_device_name(i) for i in range(len(devices))],
    "dataset_manifest_sha256": manifest_sha256,
    "adapter": (
        {"source": RUN_CONFIG["adapter_source"], "sha256": ADAPTER_SHA256} if ADAPTER else None
    ),
    "tasks_requested": len(tasks),
    "tasks_completed": len(rows),
    "stopped_by_time_guard": len(rows) < len(tasks),
    "wall_seconds": round(time.time() - STARTED, 1),
}
summary = write_run(Path(WORK) / RUN_CONFIG["run_name"], config, rows)
print(json.dumps(summary, indent=2), flush=True)
