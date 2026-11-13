"""LoRA SFT of the pinned model on the verified train demonstrations (T4, fp16 autocast).

- Installs the pinned stack and the Toolsmith wheel; RUN_CONFIG is set by kaggle/push.py.
- Builds labels with `toolsmith.sft`: only tokens the model generates at inference.
- Checks every demonstration turn against the inference prompt rendering before training.
- LoRA r=16, alpha=32, dropout 0.05 on every linear layer inside attention blocks,
  selected by full module name (so the MLP's `gate_proj` is excluded).
- AdamW (no weight decay), learning rate 2e-4 with 3 warmup steps then linear decay,
  microbatch 1, 8-example accumulation normalized by label tokens, one epoch,
  gradient checkpointing, clipping at 1.0.
- fp16 base weights, fp32 LoRA weights, dynamic loss scaling.
- Stops if the first two optimizer steps leave the adapter unchanged.
- Writes adapter/ and sft_report.json (per-step log, dev loss before/after,
  save/reload check, greedy format check) to /kaggle/working.
"""

import glob
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

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
DEVICE = "cuda:0"
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
LEARNING_RATE = 2e-4
WARMUP_STEPS = 3
ACCUMULATION = 8
MAX_GRAD_NORM = 1.0
INIT_LOSS_SCALE = 4096.0
MAX_CONTEXT_TOKENS = 4096
STARTED = time.time()

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS], check=True)
WHEEL = glob.glob("/kaggle/input/**/toolsmith_rl-*.whl", recursive=True)[0]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--ignore-requires-python", WHEEL],
    check=True,
)
DATA = os.path.dirname(WHEEL)

import importlib.metadata as metadata  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl.chat_template_utils import get_training_chat_template  # noqa: E402

from toolsmith.inference import (  # noqa: E402
    TOOL_SCHEMAS,
    ToolCallError,
    build_messages,
    parse_tool_calls,
)
from toolsmith.sft import IGNORE_INDEX, demo_messages, supervised_labels  # noqa: E402

SEED = RUN_CONFIG["seed"]
random.seed(SEED)
torch.manual_seed(SEED)
report = {
    "run": RUN_CONFIG,
    "model": MODEL,
    "model_revision": REVISION,
    "transformers_commit": TRANSFORMERS_COMMIT,
    "versions": {
        name: metadata.version(name)
        for name in ("torch", "transformers", "peft", "trl", "accelerate", "toolsmith-rl")
    },
    "gpu": torch.cuda.get_device_name(0),
}


def save_report() -> None:
    with open(os.path.join(WORK, "sft_report.json"), "w") as stream:
        json.dump(report, stream, indent=2, default=str)


tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
TEMPLATE = get_training_chat_template(tokenizer)
RENDER = {"tools": TOOL_SCHEMAS, "enable_thinking": False}
EOS = tokenizer.eos_token_id


def inference_prompt(messages: list[dict]) -> list[int]:
    """Exactly HFPolicy.reply: render with the generation prompt, then tokenize the text."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **RENDER
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def encode(messages: list[dict]) -> tuple[list[int], list[int]]:
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        chat_template=TEMPLATE,
        **RENDER,
    )
    return list(out["input_ids"]), list(out["assistant_masks"])


def rows(name: str) -> list[dict]:
    with open(os.path.join(DATA, name)) as stream:
        return [json.loads(line) for line in stream.read().splitlines()]


# The pre-filled tokens are whatever the inference prompt adds after the mask starts.
first = demo_messages(rows("demos_train.jsonl")[0])
first_ids, first_mask = encode(first[:3])
first_prompt = inference_prompt(first[:2])
THINK = first_prompt[first_mask.index(1) :]
if first_ids[: len(first_prompt)] != first_prompt or not THINK:
    raise SystemExit("Training template does not extend the inference prompt")


def examples(name: str) -> list[dict]:
    prepared = []
    for row in rows(name):
        messages = demo_messages(row)
        ids, mask = encode(messages)
        labels = supervised_labels(ids, mask, THINK, EOS)
        starts = [
            i
            for i in range(1, len(ids))
            if labels[i] != IGNORE_INDEX and labels[i - 1] == IGNORE_INDEX
        ]
        turns = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
        if len(starts) != len(turns):
            raise SystemExit(f"{row['task_id']}: {len(starts)} label spans for {len(turns)} turns")
        for turn, start in zip(turns, starts, strict=True):
            if ids[:start] != inference_prompt(messages[:turn]):
                raise SystemExit(f"{row['task_id']} turn {turn}: prefix differs from inference")
        if len(ids) > MAX_CONTEXT_TOKENS:
            raise SystemExit(f"{row['task_id']}: {len(ids)} tokens exceed the context limit")
        count = sum(label != IGNORE_INDEX for label in labels[1:])
        prepared.append({"task_id": row["task_id"], "ids": ids, "labels": labels, "count": count})
    return prepared


train, dev = examples("demos_train.jsonl"), examples("demos_dev.jsonl")
report["data"] = {
    "think_prefill": tokenizer.decode(THINK),
    "train_examples": len(train),
    "dev_examples": len(dev),
    "train_label_tokens": sum(e["count"] for e in train),
    "max_tokens": max(len(e["ids"]) for e in train + dev),
    "checked_turns_match_inference": True,
}
print(json.dumps(report["data"]), flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=REVISION, dtype=torch.float16, attn_implementation="sdpa"
).to(DEVICE)
targets = sorted(
    name
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear) and "attn" in name.lower()
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
peft_model = get_peft_model(
    model, LoraConfig(**LORA, target_modules=targets, task_type="CAUSAL_LM")
)
trainable = [(name, p) for name, p in peft_model.named_parameters() if p.requires_grad]
params = [p for _, p in trainable]
if not trainable or any("attn" not in name for name, _ in trainable):
    raise SystemExit("LoRA parameters outside attention blocks")
report["lora"] = {
    **LORA,
    "target_module_count": len(targets),
    "target_suffixes": sorted({name.split(".")[-1] for name in targets}),
    "trainable_parameters": sum(p.numel() for p in params),
    "trainable_dtypes": sorted({str(p.dtype) for p in params}),
}
print(json.dumps(report["lora"]), flush=True)


def token_nll(example: dict) -> tuple[torch.Tensor, int]:
    """Summed next-token NLL over labeled tokens (fp32 loss on fp16-autocast logits)."""
    ids = torch.tensor([example["ids"]], device=DEVICE)
    target = torch.tensor(example["labels"][1:], device=DEVICE)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = peft_model(input_ids=ids).logits
    loss = F.cross_entropy(
        logits[0, :-1].float(), target, ignore_index=IGNORE_INDEX, reduction="sum"
    )
    return loss, example["count"]


def dev_loss() -> float:
    peft_model.eval()
    total = count = 0
    with torch.no_grad():
        for example in dev:
            loss, tokens = token_nll(example)
            total, count = total + float(loss), count + tokens
    peft_model.train()
    return total / count


report["dev_loss_before"] = dev_loss()
print("dev loss before:", report["dev_loss_before"], flush=True)

order = list(range(len(train)))
random.Random(SEED).shuffle(order)
groups = [order[i : i + ACCUMULATION] for i in range(0, len(order), ACCUMULATION)]
total_steps = min(len(groups), RUN_CONFIG.get("max_steps") or len(groups))
optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.0)


def schedule(step: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    return max(0.0, (total_steps - step) / max(1, total_steps - WARMUP_STEPS))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
scaler = torch.amp.GradScaler("cuda", init_scale=INIT_LOSS_SCALE)
initial = {name: p.detach().clone() for name, p in trainable}
torch.cuda.reset_peak_memory_stats(0)
report["steps"] = []
peft_model.train()
training_started = time.time()
for step, group in enumerate(groups[:total_steps]):
    began = time.time()
    optimizer.zero_grad(set_to_none=True)
    tokens = sum(train[i]["count"] for i in group)
    summed = 0.0
    for i in group:
        loss, _ = token_nll(train[i])
        scaler.scale(loss / tokens).backward()
        summed += float(loss)
    scaler.unscale_(optimizer)
    norm = float(torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM))
    scale = scaler.get_scale()
    lr = optimizer.param_groups[0]["lr"]
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    entry = {
        "step": step + 1,
        "examples": len(group),
        "loss": summed / tokens,
        "label_tokens": tokens,
        "grad_norm": norm if math.isfinite(norm) else None,
        "lr": lr,
        "loss_scale": scale,
        "skipped": scaler.get_scale() < scale,
        "seconds": round(time.time() - began, 2),
    }
    report["steps"].append(entry)
    print(json.dumps(entry), flush=True)
    if step + 1 == 2:
        changed = sum(not torch.equal(initial[name], p.detach()) for name, p in trainable)
        applied = sum(not e["skipped"] for e in report["steps"])
        report["two_step_check"] = {"applied_steps": applied, "tensors_changed": changed}
        if applied == 0 or changed == 0:
            save_report()
            raise SystemExit(f"Two-step check failed: {report['two_step_check']}")
    save_report()
report["training_seconds"] = round(time.time() - training_started, 1)
report["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)
report["skipped_steps"] = sum(e["skipped"] for e in report["steps"])
report["tensors_changed"] = sum(not torch.equal(initial[name], p.detach()) for name, p in trainable)
report["dev_loss_after"] = dev_loss()
print("dev loss after:", report["dev_loss_after"], flush=True)

adapter_dir = os.path.join(WORK, "adapter")
peft_model.save_pretrained(adapter_dir)
with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "rb") as stream:
    report["adapter_sha256"] = hashlib.sha256(stream.read()).hexdigest()

# Fixed-input logits: trained vs base (adapter matters) and trained vs reloaded (saved).
probe = torch.tensor([dev[0]["ids"][:512]], device=DEVICE)
peft_model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    trained_logits = peft_model(input_ids=probe).logits.float()
    with peft_model.disable_adapter():
        base_logits = peft_model(input_ids=probe).logits.float()
reloaded = PeftModel.from_pretrained(peft_model.unload(), adapter_dir).to(DEVICE).eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    reloaded_logits = reloaded(input_ids=probe).logits.float()
reload_diff = float((trained_logits - reloaded_logits).abs().max())
base_diff = float((trained_logits - base_logits).abs().max())
report["reload_check"] = {
    "max_abs_logit_diff_reloaded": reload_diff,
    "max_abs_logit_diff_vs_base": base_diff,
    "ok": reload_diff < 1e-2 and base_diff > 0,
}

# Greedy first turn on three dev tasks: does the adapter still emit parseable calls?
reloaded.config.use_cache = True
samples = []
for row in rows("demos_dev.jsonl")[:3]:
    ids = torch.tensor([inference_prompt(build_messages(row["initial_context"]))], device=DEVICE)
    with torch.no_grad():
        out = reloaded.generate(
            input_ids=ids, max_new_tokens=128, do_sample=False, eos_token_id=EOS, pad_token_id=EOS
        )
    text = tokenizer.decode(out[0, ids.shape[1] :].tolist(), skip_special_tokens=False)
    text = text.split(tokenizer.eos_token)[0]
    try:
        calls, _ = parse_tool_calls(text)
        parsed = [name for name, _ in calls]
    except ToolCallError as error:
        parsed = error.reason
    samples.append({"task_id": row["task_id"], "parsed": parsed, "text": text[:400]})
report["greedy_samples"] = samples
report["wall_seconds"] = round(time.time() - STARTED, 1)
save_report()
print(json.dumps({k: report[k] for k in ("reload_check", "greedy_samples")}, indent=2))
if not report["reload_check"]["ok"]:
    raise SystemExit("Adapter save/reload check failed")
