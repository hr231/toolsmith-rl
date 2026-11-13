"""Readiness checks for the pinned model on T4 GPUs; writes /kaggle/working/gate_report.json.

- environment: record GPUs, Python, free disk, and whether Docker exists.
- install: install the pinned Transformers commit, PEFT, TRL, and the Toolsmith wheel.
- load: load the pinned model revision in fp16 with SDPA attention.
- generate: fp16 logits are finite; replies to dev tasks parse as tool calls; time
  single and batch-of-4 decoding.
- lora: LoRA forward/backward/step on a demonstration with assistant-only labels;
  gradients finite within six loss-scaled steps, weights change, save/reload
  reproduces logits.
- precision: fp16 (GPU) vs fp32 (CPU) logits on a prompt: all finite, top-1
  agreement >= 0.98, mean KL(fp32 || fp16) <= 0.02 nats, last token equal.
- The report is rewritten after every check, so partial results survive a crash.
"""

import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

MODEL = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
REVISION = "f6c832e6963ddb74f62c75813a3e8c74b1af67b1"
TRANSFORMERS = "transformers @ git+https://github.com/shan18/transformers@fcc6924dfbac964f00787859b568c867ef5ad4cb"
PINS = [TRANSFORMERS, "peft==0.21.2", "trl==1.17.1"]
REPORT = "/kaggle/working/gate_report.json"
SEED = 17
RUN_CONFIG = {}  # replaced by kaggle/push.py
report = {
    "run": RUN_CONFIG,
    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "gates": {},
}


def save():
    with open(REPORT, "w") as stream:
        json.dump(report, stream, indent=2, default=str)


def gate(name):
    def wrap(function):
        started = time.time()
        try:
            details = function()
            report["gates"][name] = {"ok": bool(details.pop("ok", True)), **details}
        except Exception:
            report["gates"][name] = {"ok": False, "error": traceback.format_exc()[-4000:]}
        report["gates"][name]["seconds"] = round(time.time() - started, 1)
        save()
        print(name, json.dumps(report["gates"][name], default=str)[:2000], flush=True)
        return report["gates"][name]["ok"]

    return wrap


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return {"returncode": result.returncode, "out": (result.stdout + result.stderr)[-3000:]}


@gate("environment")
def environment():
    return {
        "python": platform.python_version(),
        "nvidia_smi": run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"]
        ),
        "docker_binary": shutil.which("docker"),
        "docker_info": run(["docker", "info"])
        if shutil.which("docker")
        else "docker not installed",
        "disk_free_gb": round(shutil.disk_usage("/kaggle/working").free / 1e9, 1),
    }


WHEELS = glob.glob("/kaggle/input/**/toolsmith_rl-*.whl", recursive=True)

if not WHEELS:
    report["gates"]["install"] = {
        "ok": False,
        "error": "toolsmith wheel not found under /kaggle/input",
    }
    report["passed"] = False
    save()
    raise SystemExit("toolsmith wheel not found; attach the public dataset")


@gate("install")
def install():
    # Remove the image's torchao: PEFT 0.21.2 rejects old torchao versions at LoRA injection.
    removed = run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
    pins = run([sys.executable, "-m", "pip", "install", "-q", *PINS])
    ours = run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "--ignore-requires-python",
            WHEELS[0],
        ]
    )
    import importlib.metadata as metadata

    versions = {
        name: metadata.version(name)
        for name in ("torch", "transformers", "peft", "trl", "accelerate", "toolsmith-rl")
    }
    return {
        "ok": pins["returncode"] == 0 and ours["returncode"] == 0,
        "versions": versions,
        "pip": pins["out"][-1500:],
        "torchao_uninstall": removed["out"][-300:],
        "wheel": WHEELS[0],
    }


import torch  # noqa: E402

DATA = os.path.dirname(WHEELS[0])
state = {}


def public_rows(name, limit):
    with open(os.path.join(DATA, name)) as stream:
        return [json.loads(line) for line in stream.read().splitlines()[:limit]]


@gate("load")
def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda:0")
    model.eval()
    state.update(tokenizer=tokenizer, model=model)
    config = model.config
    return {
        "ok": config.n_recurrent_loops == 2 and config.recurrent_kv_cache_mode == "per_loop",
        "load_seconds": round(time.time() - started, 1),
        "parameters": sum(p.numel() for p in model.parameters()),
        "loops": config.n_recurrent_loops,
        "kv_cache_mode": config.recurrent_kv_cache_mode,
        "gpu_gb_allocated": round(torch.cuda.memory_allocated(0) / 1e9, 2),
    }


def prompt_ids(task_row):
    from toolsmith.episode import Episode
    from toolsmith.inference import TOOL_SCHEMAS, build_messages
    from toolsmith.tasks import BugTask

    context = Episode(BugTask.from_dict(task_row)).initial_context
    text = state["tokenizer"].apply_chat_template(
        build_messages(context),
        tools=TOOL_SCHEMAS,
        enable_thinking=False,
        add_generation_prompt=True,
        tokenize=False,
    )
    return state["tokenizer"](text, return_tensors="pt", add_special_tokens=False)["input_ids"]


@gate("generate")
def generate():
    from toolsmith.inference import ToolCallError, parse_tool_calls

    tokenizer, model = state["tokenizer"], state["model"]
    samples, finite = [], True
    for row in public_rows("dev.jsonl", 3):
        ids = prompt_ids(row).to("cuda:0")
        with torch.no_grad():
            logits = model(ids).logits
        finite &= bool(torch.isfinite(logits).all())
        torch.manual_seed(SEED)
        started = time.time()
        with torch.no_grad():
            output = model.generate(
                ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
            )
        new = output[0, ids.shape[1] :]
        text = tokenizer.decode(new, skip_special_tokens=False).split(tokenizer.eos_token)[0]
        try:
            calls, _ = parse_tool_calls(text)
            parsed = {"tools": [name for name, _ in calls]}
        except ToolCallError as error:
            parsed = {"error": error.reason}
        samples.append(
            {
                "task_id": row["task_id"],
                "prompt_tokens": ids.shape[1],
                "new_tokens": len(new),
                "seconds": round(time.time() - started, 2),
                "parsed": parsed,
                "text": text[:600],
            }
        )
    parsed_ok = sum("tools" in s["parsed"] for s in samples)
    # Batch-of-4 decoding throughput (GRPO samples G completions per prompt).
    batch = prompt_ids(public_rows("dev.jsonl", 1)[0]).to("cuda:0").repeat(4, 1)
    torch.manual_seed(SEED)
    started = time.time()
    with torch.no_grad():
        output = model.generate(
            batch,
            max_new_tokens=128,
            min_new_tokens=128,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
        )
    batch_seconds = time.time() - started
    generated = (output.shape[1] - batch.shape[1]) * batch.shape[0]
    return {
        "ok": finite and parsed_ok >= 1,
        "finite_logits": finite,
        "replies_parsed": parsed_ok,
        "single_tokens_per_second": round(
            sum(s["new_tokens"] for s in samples) / sum(s["seconds"] for s in samples), 2
        ),
        "batch4_tokens_per_second": round(generated / batch_seconds, 2),
        "samples": samples,
    }


@gate("lora")
def lora():
    from peft import LoraConfig, PeftModel, get_peft_model
    from trl.chat_template_utils import get_training_chat_template

    from toolsmith.inference import TOOL_SCHEMAS, build_messages

    tokenizer, model = state["tokenizer"], state["model"]
    linear_names = sorted(
        {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
    )
    attention = sorted(
        {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and "attn" in name.lower()
        }
    )
    demo = public_rows("demos_train.jsonl", 1)[0]
    messages = build_messages(demo["initial_context"])
    for event in demo["events"]:
        call = {
            "type": "function",
            "function": {"name": event["tool"], "arguments": event["arguments"]},
        }
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "content": json.dumps(event["observation"])},
        ]
    encoded = tokenizer.apply_chat_template(
        messages[:-1],
        tools=TOOL_SCHEMAS,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        chat_template=get_training_chat_template(tokenizer),
    )
    ids = torch.tensor([encoded["input_ids"]], device="cuda:0")
    labels = ids.clone()
    labels[0, torch.tensor(encoded["assistant_masks"], device="cuda:0") == 0] = -100
    config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=attention, task_type="CAUSAL_LM"
    )
    try:
        model.gradient_checkpointing_enable()
        checkpointing = True
    except (ValueError, NotImplementedError) as error:
        checkpointing = f"unsupported: {error}"
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, config)
    peft_model.train()
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats(0)
    # Dynamic fp16 loss scaling skips overflowing steps; require one clean step within six.
    steps, started = [], time.time()
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = peft_model(input_ids=ids, labels=labels).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite = all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        steps.append({"loss": round(float(loss), 4), "grads_finite": bool(finite), "scale": scale})
        if finite:
            break
    step_seconds = (time.time() - started) / len(steps)
    grads_finite = steps[-1]["grads_finite"]
    changed = sum(not torch.equal(b, p.detach()) for b, p in zip(before, trainable, strict=True))
    peft_model.eval()
    probe = ids[:, :256]
    with torch.no_grad():
        trained_logits = peft_model(probe).logits.float()
    peft_model.save_pretrained("/kaggle/working/gate_adapter")
    base = peft_model.unload()
    reloaded = PeftModel.from_pretrained(base, "/kaggle/working/gate_adapter").to("cuda:0").eval()
    with torch.no_grad():
        reloaded_logits = reloaded(probe).logits.float()
    reload_max_diff = float((trained_logits - reloaded_logits).abs().max())
    state["model"] = reloaded.unload()
    if checkpointing is True:
        state["model"].gradient_checkpointing_disable()
    return {
        "ok": bool(torch.isfinite(loss))
        and grads_finite
        and changed > 0
        and reload_max_diff < 1e-2,
        "linear_module_names": linear_names,
        "lora_targets": attention,
        "gradient_checkpointing": checkpointing,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "sequence_tokens": ids.shape[1],
        "assistant_label_tokens": int((labels != -100).sum()),
        "loss": float(loss),
        "steps": steps,
        "grads_finite": grads_finite,
        "tensors_changed": changed,
        "step_seconds": round(step_seconds, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
        "reload_max_logit_diff": reload_max_diff,
    }


@gate("precision")
def precision():
    import gc

    from transformers import AutoModelForCausalLM

    ids = prompt_ids(public_rows("dev.jsonl", 1)[0])
    with torch.no_grad():
        half = state["model"](ids.to("cuda:0")).logits.float().cpu()[0]
    state["model"].to("cpu")
    del state["model"]
    gc.collect()
    torch.cuda.empty_cache()
    started = time.time()
    full_model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    with torch.no_grad():
        full = full_model(ids).logits.float()[0]
    log_full, log_half = full.log_softmax(-1), half.log_softmax(-1)
    kl = (log_full.exp() * (log_full - log_half)).sum(-1)
    agreement = float((full.argmax(-1) == half.argmax(-1)).float().mean())
    finite = bool(torch.isfinite(half).all() and torch.isfinite(full).all())
    last_equal = bool(full[-1].argmax() == half[-1].argmax())
    return {
        "ok": finite and agreement >= 0.98 and float(kl.mean()) <= 0.02 and last_equal,
        "positions": ids.shape[1],
        "finite": finite,
        "top1_agreement": agreement,
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
        "last_token_equal": last_equal,
        "max_abs_logit_diff": float((full - half).abs().max()),
        "fp32_cpu_seconds": round(time.time() - started, 1),
    }


report["passed"] = all(gate_result["ok"] for gate_result in report["gates"].values())
report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
print("GATE PASSED" if report["passed"] else "GATE FAILED", flush=True)
