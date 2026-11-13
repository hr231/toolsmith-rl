"""Render a notebook script with its run settings and push it to run on T4 GPUs.

Subcommands:
- gate: load, generation, LoRA, and fp16-precision checks for the pinned model.
- eval: evaluate the model, optionally with an adapter from another notebook's output.
- sft: LoRA SFT on the verified train demonstrations.
- grpo: up to two GRPO runs (base or SFT start), one per GPU.

Steps:
- Replace the script's RUN_CONFIG marker with the settings as a Python literal.
- Compile the rendered script, then write it and its notebook metadata to build/.
- Attach the public dataset, the sandbox-token dataset (eval, grpo), and the adapter
  notebook output (when given); the token itself is never written into the script.
- Push with `kaggle kernels push`.

Usage:
  python kaggle/push.py eval --kaggle-user <user> --kernel toolsmith-dory-baseline \\
      --run-name dory-base-full --url https://<tunnel> --splits dev test_iid
  python kaggle/push.py sft --kaggle-user <user> --kernel toolsmith-dory-sft \\
      --run-name dory-sft-seed17
"""

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = {
    "gate": HERE / "gate" / "gate.py",
    "eval": HERE / "evaluate" / "evaluate.py",
    "sft": HERE / "sft" / "sft.py",
    "grpo": HERE / "grpo" / "grpo.py",
}
MARKER = "RUN_CONFIG = {}  # replaced by kaggle/push.py"
SPLITS = {"train", "dev", "test_iid", "test_compositional", "challenge"}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="kind", required=True)
    gate = commands.add_parser("gate")
    evaluate = commands.add_parser("eval")
    evaluate.add_argument("--url", required=True)
    evaluate.add_argument("--splits", nargs="+", required=True)
    evaluate.add_argument("--limit", type=int, default=10_000, help="Tasks per split.")
    source = evaluate.add_mutually_exclusive_group()
    source.add_argument("--adapter-kernel", help="Notebook slug whose output holds the adapter.")
    source.add_argument("--adapter-dataset", help="Private dataset slug holding the adapter.")
    evaluate.add_argument(
        "--adapter-dir",
        default="adapter",
        help="Directory holding adapter_config.json, e.g. grpo-base/adapter or the dataset slug.",
    )
    evaluate.add_argument("--shard", help="I/N: evaluate every N-th task starting at task I.")
    sft = commands.add_parser("sft")
    sft.add_argument("--max-steps", type=int, help="Stop after this many optimizer steps.")
    grpo = commands.add_parser("grpo")
    grpo.add_argument("--url", required=True)
    grpo.add_argument(
        "--start", nargs="+", choices=["base", "sft"], required=True, help="One run per GPU."
    )
    grpo.add_argument("--adapter-kernel", help="Kernel slug of the SFT run (for --start sft).")
    grpo.add_argument("--max-steps", type=int, required=True)
    grpo.add_argument("--learning-rate", type=float, default=2e-5)
    grpo.add_argument("--hours", type=float, default=8.0, help="Training budget per run.")
    grpo.add_argument("--save-every", type=int, default=25, help="Adapter checkpoint interval.")
    for command in (gate, evaluate, sft, grpo):
        command.add_argument("--kaggle-user", required=True)
        command.add_argument(
            "--kernel", required=True, help="Kernel slug, e.g. toolsmith-dory-sft."
        )
        command.add_argument("--run-name", required=True)
        command.add_argument("--seed", type=int, default=17)
        command.add_argument("--timeout", type=int, default=43_000, help="Seconds.")
        command.add_argument("--no-run", action="store_true", help="Save without running.")
    args = parser.parse_args()

    config = {"run_name": args.run_name, "seed": args.seed}
    datasets = [f"{args.kaggle_user}/toolsmith-repair-public"]
    kernels = []
    if args.kind in {"eval", "grpo"}:
        if not args.url.startswith("https://"):
            raise SystemExit("Tunnel URL must be HTTPS")
        config["sandbox_url"] = args.url
        datasets.append(f"{args.kaggle_user}/toolsmith-sandbox-token")
        if args.adapter_kernel:
            kernels.append(f"{args.kaggle_user}/{args.adapter_kernel}")
    if args.kind == "eval":
        if unknown := set(args.splits) - SPLITS:
            raise SystemExit(f"Unknown splits: {sorted(unknown)}")
        adapter_source = None
        if args.adapter_kernel:
            adapter_source = f"kernel:{args.adapter_kernel}"
        elif args.adapter_dataset:
            adapter_source = f"dataset:{args.adapter_dataset}"
            datasets.append(f"{args.kaggle_user}/{args.adapter_dataset}")
        config |= {
            "splits": args.splits,
            "limit": args.limit,
            "adapter_source": adapter_source,
            "adapter_dir": args.adapter_dir,
        }
        if args.shard:
            index, count = (int(part) for part in args.shard.split("/"))
            if not 0 <= index < count:
                raise SystemExit("--shard must be I/N with 0 <= I < N")
            config["shard"] = [index, count]
    elif args.kind == "grpo":
        if len(args.start) > 2 or len(set(args.start)) != len(args.start):
            raise SystemExit("At most two distinct starts (one per T4)")
        if "sft" in args.start and not args.adapter_kernel:
            raise SystemExit("--start sft needs --adapter-kernel")
        config["adapter_kernel"] = args.adapter_kernel
        config["runs"] = [
            {
                "name": f"grpo-{start}",
                "start": start,
                "max_steps": args.max_steps,
                "learning_rate": args.learning_rate,
                "hours": args.hours,
                "save_every": args.save_every,
            }
            for start in args.start
        ]
    elif args.kind == "sft":
        config["max_steps"] = args.max_steps

    script = SCRIPTS[args.kind]
    source = script.read_text()
    if source.count(MARKER) != 1:
        raise SystemExit(f"{script} RUN_CONFIG marker not found exactly once")
    output = Path("build/kaggle-kernels") / args.kernel
    output.mkdir(parents=True, exist_ok=True)
    # Python literal, not JSON (None/True rather than null/true); compile before writing.
    rendered = source.replace(MARKER, f"RUN_CONFIG = {config!r}")
    compile(rendered, script.name, "exec")
    (output / script.name).write_text(rendered)
    metadata = {
        "id": f"{args.kaggle_user}/{args.kernel}",
        "title": args.kernel,
        "code_file": script.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": kernels,
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(output), "--accelerator", "NvidiaTeslaT4"]
        + ["-t", str(args.timeout)]
        + (["--no-run"] if args.no_run else []),
        check=True,
    )


if __name__ == "__main__":
    main()
