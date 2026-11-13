"""Lightweight CLI that does not import optional ML dependencies."""

import argparse
import json
import platform
import sqlite3
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from toolsmith.routing import AttemptOutcome, RoutingPolicy, load_policy, route_task
from toolsmith.tasks import load_task


def environment_report() -> dict[str, object]:
    """Inventory this interpreter without claiming accelerator readiness."""
    packages = {}
    for name in ("torch", "transformers", "peft", "datasets", "trl"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    return {
        "toolsmith": version("toolsmith-rl"),
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "sqlite": sqlite3.sqlite_version,
        "ml_packages": packages,
        "scope": "Inventory only; accelerator availability and training readiness are not checked.",
    }


def main() -> None:
    """Run the installed command-line entry point."""
    parser = argparse.ArgumentParser(
        prog="toolsmith",
        description="Python bug-repair experiments: routing, restricted execution, and evaluation.",
    )
    parser.add_argument("--version", action="version", version=version("toolsmith-rl"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Print environment inventory as JSON; no model downloads.")
    route = commands.add_parser("route", help="Inspect a routing decision; does not run models.")
    route.add_argument("--task", type=Path, required=True, help="Public bug-task JSON manifest.")
    route.add_argument(
        "--policy", type=Path, help="Explicit router TOML; otherwise use built-in limits."
    )
    outcome = route.add_mutually_exclusive_group()
    outcome.add_argument("--after-small", type=AttemptOutcome, choices=list(AttemptOutcome))
    outcome.add_argument("--after-large", type=AttemptOutcome, choices=list(AttemptOutcome))
    route.add_argument(
        "--allow-large",
        action="store_true",
        help="Allow selecting the large-model route; does not authorize or make API calls.",
    )
    sandbox_check = commands.add_parser(
        "sandbox-check", help="Probe a real local sandbox container."
    )
    evaluate = commands.add_parser(
        "evaluate-patch", help="Evaluate a source patch in fresh containers."
    )
    evaluate.add_argument("--task", type=Path, required=True)
    evaluate.add_argument(
        "--patch", type=Path, required=True, help="Strict unified source-file diff."
    )
    evaluate.add_argument(
        "--checks", type=Path, required=True, help="Private evaluator JSON manifest."
    )
    evaluate.add_argument(
        "--output", type=Path, required=True, help="New directory for result artifacts."
    )
    data = commands.add_parser("data", help="Generate, validate, and demonstrate repair datasets.")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    generate = data_commands.add_parser("generate", help="Write a new deterministic dataset.")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    validate = data_commands.add_parser(
        "validate", help="Check by regeneration; --execute also runs every task in Docker."
    )
    validate.add_argument("--directory", type=Path, required=True)
    validate.add_argument("--execute", action="store_true")
    validate.add_argument("--output", type=Path, help="New report directory for --execute.")
    demonstrations = data_commands.add_parser(
        "demonstrations", help="Build replay-verified train/dev reference trajectories."
    )
    demonstrations.add_argument("--directory", type=Path, required=True)
    demonstrations.add_argument("--output", type=Path, required=True)
    for command in (validate, demonstrations):
        command.add_argument("--workers", type=int, default=4, help="Parallel containers, 1-8.")
    serve = commands.add_parser(
        "sandbox-serve",
        help="Serve the Docker sandbox on 127.0.0.1 (expose only via an authenticated tunnel).",
    )
    serve.add_argument("--directory", type=Path, required=True, help="Validated dataset.")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--workers", type=int, default=4, help="Concurrent sandbox runs, 1-16.")
    serve.add_argument(
        "--token-file",
        type=Path,
        default=Path.home() / ".toolsmith" / "sandbox_token",
        help="Private bearer-token file; created with mode 0600 if missing.",
    )
    serve.add_argument(
        "--score-log", type=Path, required=True, help="JSONL log of full score results."
    )
    serve.add_argument(
        "--rotate-token", action="store_true", help="Replace the token file before starting."
    )
    compare = commands.add_parser(
        "compare", help="Compare recorded runs on identical frozen tasks (paired bootstrap)."
    )
    compare.add_argument(
        "--runs", nargs="+", required=True, help="NAME=RUN_DIR; the first is the reference."
    )
    compare.add_argument("--output", type=Path, required=True, help="New report directory.")
    compare.add_argument("--seed", type=int, default=17)
    compare.add_argument("--resamples", type=int, default=2000)
    view = commands.add_parser("view", help="Write a static, read-only HTML viewer over runs.")
    view.add_argument("--runs", nargs="+", required=True, help="NAME=RUN_DIR, in column order.")
    view.add_argument("--tasks", type=Path, required=True, help="Validated dataset directory.")
    view.add_argument("--policy", type=Path, help="Router TOML; otherwise built-in limits.")
    view.add_argument("--output", type=Path, required=True, help="New site directory.")
    merge = commands.add_parser(
        "merge-runs", help="Combine shard runs of one model into a single run directory."
    )
    merge.add_argument("--shards", type=Path, nargs="+", required=True, help="Shard run dirs.")
    merge.add_argument("--name", required=True, help="Run name for the merged run.")
    merge.add_argument("--output", type=Path, required=True, help="New run directory.")
    for command in (sandbox_check, evaluate, validate, demonstrations, serve):
        command.add_argument(
            "--image", default="toolsmith-sandbox:py312-v1", help="Existing local image."
        )
        command.add_argument(
            "--timeout", type=float, default=10.0, help="Wall seconds per container."
        )
        command.add_argument(
            "--output-limit", type=int, default=65536, help="Combined output byte cap."
        )
    args = parser.parse_args()
    if args.command == "doctor":
        print(json.dumps(environment_report(), indent=2))
    elif args.command == "route":
        try:
            task = load_task(args.task)
            policy = load_policy(args.policy) if args.policy else RoutingPolicy()
            decision = route_task(
                task,
                policy,
                after_small=args.after_small,
                after_large=args.after_large,
                allow_large=args.allow_large,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(
            json.dumps(
                {
                    "mode": "decision_only",
                    "task_id": task.task_id,
                    "features": {
                        "source_file_count": task.source_file_count,
                        "input_bytes": task.input_bytes,
                    },
                    "policy": asdict(policy),
                    **asdict(decision),
                },
                indent=2,
            )
        )
    elif args.command in ("sandbox-check", "evaluate-patch"):
        from toolsmith.sandbox import SandboxError, SandboxLimits

        try:
            limits = SandboxLimits(timeout_seconds=args.timeout, output_bytes=args.output_limit)
            if args.command == "sandbox-check":
                from toolsmith.diagnostics import check_sandbox

                result = check_sandbox(image=args.image, limits=limits)
            else:
                from toolsmith.evaluator import evaluate_patch, load_checks, write_evaluation

                if args.output.exists():
                    raise ValueError(f"Output directory already exists: {args.output}")
                task = load_task(args.task)
                patch = args.patch.read_text(encoding="utf-8")
                checks = load_checks(args.checks)
                result = evaluate_patch(task, patch, checks, image=args.image, limits=limits)
                write_evaluation(args.output, result, task, patch)
        except SandboxError as error:
            result = {"success": False, "status": "infrastructure_error", "error": str(error)}
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2, allow_nan=False))
        if not result["success"]:
            raise SystemExit(1)
    elif args.command == "data":
        _data_command(args, parser)
    elif args.command == "sandbox-serve":
        _serve_command(args, parser)
    elif args.command in ("compare", "view"):
        _runs_command(args, parser)
    elif args.command == "merge-runs":
        from toolsmith.evaluation import write_run
        from toolsmith.reporting import merge_runs

        try:
            config, rows = merge_runs(args.shards, args.name)
            summary = write_run(args.output, config, rows)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        print(json.dumps({k: summary[k] for k in ("episodes", "success_rate")}, indent=2))


def _runs_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from toolsmith.reporting import compare, load_run, split_of, write_report

    try:
        runs = []
        for spec in args.runs:
            name, separator, path = spec.partition("=")
            if not separator or not name:
                raise ValueError(f"Expected NAME=RUN_DIR, got {spec!r}")
            runs.append(load_run(Path(path), name))
        if args.command == "compare":
            report = compare(runs, seed=args.seed, resamples=args.resamples)
            print(write_report(args.output, report), end="")
            return
        from toolsmith.dataset import load_records
        from toolsmith.viewer import build_site

        splits = sorted({split_of(task_id) for task_id in runs[0].rows})
        tasks = {r.task.task_id: r.task for s in splits for r in load_records(args.tasks, s)}
        policy = load_policy(args.policy) if args.policy else RoutingPolicy()
        index = build_site(runs, tasks, policy, args.output)
        print(index)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


def _serve_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from toolsmith.remote import (
        SandboxService,
        ensure_token,
        load_service_records,
        run_service_forever,
    )
    from toolsmith.sandbox import SandboxLimits

    try:
        if args.rotate_token:
            args.token_file.unlink(missing_ok=True)
        service = SandboxService(
            load_service_records(args.directory),
            token=ensure_token(args.token_file),
            image=args.image,
            limits=SandboxLimits(timeout_seconds=args.timeout, output_bytes=args.output_limit),
            workers=args.workers,
            score_log=args.score_log,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Serving sandbox on http://127.0.0.1:{args.port} (token in {args.token_file}). "
        "Expose it only through an authenticated HTTPS tunnel. Ctrl-C to stop.",
        flush=True,
    )
    try:
        run_service_forever(service, port=args.port)
    except KeyboardInterrupt:
        pass


def _data_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from toolsmith.dataset import generate_dataset, load_config, validate_dataset
    from toolsmith.sandbox import SandboxError, SandboxLimits

    try:
        if args.data_command == "generate":
            result = generate_dataset(load_config(args.config), args.output)
            print(json.dumps({"success": True, **result}, indent=2))
            return
        limits = SandboxLimits(timeout_seconds=args.timeout, output_bytes=args.output_limit)
        if args.data_command == "validate" and not args.execute:
            if args.output is not None:
                raise ValueError("--output applies only with --execute")
            result = {
                "success": True,
                "execution_verified": False,
                **validate_dataset(args.directory),
            }
        elif args.data_command == "validate":
            from toolsmith.reference import verify_dataset

            if args.output is None:
                raise ValueError("--execute requires a new --output report directory")
            result = verify_dataset(
                args.directory, args.output, image=args.image, limits=limits, workers=args.workers
            )
        else:
            from toolsmith.reference import build_demonstrations

            result = build_demonstrations(
                args.directory, args.output, image=args.image, limits=limits, workers=args.workers
            )
    except SandboxError as error:
        result = {"success": False, "status": "infrastructure_error", "error": str(error)}
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
