"""Stateful GRPO of the pinned model on T4 GPUs, one training run per GPU.

- RUN_CONFIG (set by kaggle/push.py): sandbox URL, seed, and up to two runs, each with
  a start ("base": fresh LoRA on the prompt-only model; "sft": keep training the SFT
  adapter), step cap, learning rate, wall-clock budget, and checkpoint interval.
- Parent process: installs the pinned stack once, then starts one worker per run with
  a single visible GPU (GRPOTrainer rejects nn.DataParallel).
- Worker: rollouts come from `toolsmith.grpo` through TRL's `rollout_func`; each group
  is G=4 independent episodes of one train task, generated in lockstep and scored by
  the remote sandbox service.
- TRL keeps a frozen "ref" copy of the starting adapter as the KL reference (beta 0.02).
- Microbatch 1 with 4-step accumulation (one group per optimizer step), gradient
  checkpointing, fp16 autocast, dropout disabled to match eval-mode rollouts.
- Before training: checks that batched (left-padded) decoding matches single-row
  decoding (rollouts are batched only if so) and that the folded LM head reproduces
  the model's log-probs.
- After training: checks that the trained adapter changed, the ref adapter equals the
  starting weights, and sampled base weights are unchanged.
- Writes /kaggle/working/<run>/: adapter/, checkpoint-<step>/, grpo_rollouts.jsonl,
  grpo_report.json.
"""

import glob
import hashlib
import json
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
NUM_GENERATIONS = 4
BETA = 0.02
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}  # same as SFT
STARTED = time.time()

TOKEN_FILES = glob.glob("/kaggle/input/**/sandbox_token.txt", recursive=True)
if len(TOKEN_FILES) != 1:
    raise SystemExit(f"Expected one sandbox_token.txt in inputs, found {len(TOKEN_FILES)}")
with open(TOKEN_FILES[0]) as stream:
    SANDBOX_TOKEN = stream.read().strip()

if "--worker" not in sys.argv:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS], check=True)
    wheel = glob.glob("/kaggle/input/**/toolsmith_rl-*.whl", recursive=True)[0]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--ignore-requires-python"]
        + [wheel],
        check=True,
    )
    # Start workers one at a time: each waits for <run>/loaded so two model loads
    # never hold host RAM at once.
    workers = []
    for index, run in enumerate(RUN_CONFIG["runs"]):
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(index), "TRL_EXPERIMENTAL_SILENCE": "1"}
        worker = subprocess.Popen([sys.executable, __file__, "--worker", str(index)], env=env)
        workers.append(worker)
        marker = os.path.join(WORK, run["name"], "loaded")
        while not os.path.exists(marker) and worker.poll() is None:
            time.sleep(5)
    runs = zip(RUN_CONFIG["runs"], workers, strict=True)
    codes = {run["name"]: worker.wait() for run, worker in runs}
    with open(os.path.join(WORK, "grpo_runs.json"), "w") as stream:
        json.dump({"exit_codes": codes, "wall_seconds": round(time.time() - STARTED, 1)}, stream)
    print("worker exit codes:", codes, flush=True)
    if any(codes.values()):
        raise SystemExit(f"GRPO worker failed: {codes}")
    raise SystemExit(0)

RUN = RUN_CONFIG["runs"][int(sys.argv[sys.argv.index("--worker") + 1])]
OUT = os.path.join(WORK, RUN["name"])
os.makedirs(OUT, exist_ok=True)
STOP_AT = STARTED + RUN["hours"] * 3600
DEVICE = "cuda:0"
DATA = os.path.dirname(glob.glob("/kaggle/input/**/toolsmith_rl-*.whl", recursive=True)[0])

import importlib.metadata as metadata  # noqa: E402

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from toolsmith.episode import Episode  # noqa: E402
from toolsmith.evaluation import GenerationConfig  # noqa: E402
from toolsmith.grpo import (  # noqa: E402
    ChatRenderer,
    Generated,
    episode_reward,
    make_rollout_func,
    training_prompts,
)
from toolsmith.inference import build_messages, render_tool_call  # noqa: E402
from toolsmith.remote import RemoteClient  # noqa: E402
from toolsmith.tasks import BugTask  # noqa: E402


def log(*parts) -> None:
    print(f"[{RUN['name']}]", *parts, flush=True)


SEED = RUN_CONFIG["seed"]
random.seed(SEED)
torch.manual_seed(SEED)
report = {
    "run": RUN,
    "seed": SEED,
    "sandbox_url": RUN_CONFIG["sandbox_url"],
    "model": MODEL,
    "model_revision": REVISION,
    "versions": {
        name: metadata.version(name)
        for name in ("torch", "transformers", "peft", "trl", "accelerate", "toolsmith-rl")
    },
    "gpu": torch.cuda.get_device_name(0),
}


def save_report() -> None:
    with open(os.path.join(OUT, "grpo_report.json"), "w") as stream:
        json.dump(report, stream, indent=2, default=str)


client = RemoteClient(RUN_CONFIG["sandbox_url"], SANDBOX_TOKEN)
log("sandbox health:", client.health())
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
EOS = tokenizer.eos_token_id
PAD = tokenizer.pad_token_id
render = ChatRenderer(tokenizer)
generation = GenerationConfig(seed=SEED)

base = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=REVISION, dtype=torch.float16, attn_implementation="sdpa"
).to(DEVICE)
with open(os.path.join(OUT, "loaded"), "w") as stream:
    stream.write("ok\n")

# TRL's fused LM head computes hidden @ lm_head.weight and skips DoryLMHead's learned
# per-token logit scale. Fold that scale into an untied copy of the head weight and set
# the remaining factor to 1, so the model's forward and TRL's fused head compute the
# same logits; the logit check below verifies both against the original model.
with open(os.path.join(DATA, "dev.jsonl")) as stream:
    first_dev = BugTask.from_dict(json.loads(stream.readline()))
probe_reply = tokenizer(
    render_tool_call("read_file", {"path": "src/solution.py"}), add_special_tokens=False
)["input_ids"] + [EOS]
LOGIT_PROBE = torch.tensor(
    [render.prompt_ids(build_messages(Episode(first_dev).initial_context)) + probe_reply],
    device=DEVICE,
)
PROBE_TOKENS = len(probe_reply)


def forward_logprobs(net) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        logits = net(input_ids=LOGIT_PROBE).logits.float()
    logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
    return logprobs.gather(1, LOGIT_PROBE[0, 1:, None])[-PROBE_TOKENS:, 0]


base.eval()
original_logprobs = forward_logprobs(base)
head = base.get_output_embeddings()
scale = head.logit_scale.detach().float() * head.logit_scale_factor
head.weight = torch.nn.Parameter(
    (head.weight.detach().float() * scale[:, None]).to(head.weight.dtype), requires_grad=False
)
head.logit_scale = torch.nn.Parameter(
    torch.full_like(scale, 1.0 / head.logit_scale_factor), requires_grad=False
)
base.config.tie_word_embeddings = False
report["logit_scale"] = {
    "min": float(scale.min()),
    "max": float(scale.max()),
    "mean": float(scale.mean()),
}

if RUN["start"] == "sft":
    adapters = glob.glob("/kaggle/input/**/adapter/adapter_config.json", recursive=True)
    if len(adapters) != 1:
        raise SystemExit(f"Expected one SFT adapter in inputs, found {adapters}")
    sft_dir = os.path.dirname(adapters[0])
    with open(os.path.join(sft_dir, "adapter_model.safetensors"), "rb") as stream:
        report["start_adapter_sha256"] = hashlib.sha256(stream.read()).hexdigest()
    model = PeftModel.from_pretrained(base, sft_dir, is_trainable=True).to(DEVICE)
elif RUN["start"] == "base":
    targets = sorted(
        name
        for name, module in base.named_modules()
        if isinstance(module, torch.nn.Linear) and "attn" in name.lower()
    )
    config = LoraConfig(**LORA, target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(base, config).to(DEVICE)
else:
    raise SystemExit(f"Unknown start: {RUN['start']}")
for param in model.parameters():
    if param.requires_grad and param.dtype != torch.float32:
        param.data = param.data.float()  # fp16 AMP needs fp32 trainable weights
model.enable_input_require_grads()
trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
if not trainable or any("attn" not in n for n in trainable):
    raise SystemExit("Trainable parameters outside attention LoRA")
initial = {n: p.detach().clone() for n, p in trainable.items()}
base_sample = [n for n, p in model.named_parameters() if not p.requires_grad][::97][:6]
base_before = {n: model.get_parameter(n).detach().float().sum().item() for n in base_sample}
report["trainable"] = {
    "tensors": len(trainable),
    "parameters": sum(p.numel() for p in trainable.values()),
    "dtypes": sorted({str(p.dtype) for p in trainable.values()}),
}


def generate_rows(rows: list[list[int]], *, sample: bool, max_new_tokens: int) -> list[list[int]]:
    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), PAD, dtype=torch.long)
    attention = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        ids[index, width - len(row) :] = torch.tensor(row)
        attention[index, width - len(row) :] = 1
    options = {"do_sample": True, "temperature": 1.0, "top_p": 0.95} if sample else {}
    with torch.no_grad():
        output = model.generate(
            input_ids=ids.to(DEVICE),
            attention_mask=attention.to(DEVICE),
            max_new_tokens=max_new_tokens,
            eos_token_id=EOS,
            pad_token_id=PAD,
            use_cache=True,
            **({"do_sample": False} | options),
        )
    return [output[index, width:].tolist() for index in range(len(rows))]


# Left-padding check: greedy batched rows must match greedy single rows.
model.eval()
with open(os.path.join(DATA, "dev.jsonl")) as stream:
    probe_tasks = [BugTask.from_dict(json.loads(line)) for line in stream.read().splitlines()[:2]]
probe_prompts = [render.prompt_ids(build_messages(Episode(t).initial_context)) for t in probe_tasks]
probe_prompts[1] = probe_prompts[1] + render.tool_suffix_ids(['{"content":"x = 1\\n"}'])
batched_rows = generate_rows(probe_prompts, sample=False, max_new_tokens=24)
single_rows = [generate_rows([p], sample=False, max_new_tokens=24)[0] for p in probe_prompts]
BATCHED = all(
    b[: s.index(EOS) + 1 if EOS in s else len(s)] == s[: s.index(EOS) + 1 if EOS in s else len(s)]
    for b, s in zip(batched_rows, single_rows, strict=True)
)
report["padding_check"] = {
    "prompt_lengths": [len(p) for p in probe_prompts],
    "batched_equals_single": BATCHED,
    "batched": [tokenizer.decode(r) for r in batched_rows],
    "single": [tokenizer.decode(r) for r in single_rows],
}
log(json.dumps(report["padding_check"]))
save_report()


def generate(contexts: dict[int, list[int]], seed: int) -> dict[int, Generated]:
    """One rollout turn: sample every active episode (batched when padding is safe)."""
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)
    keys = list(contexts)
    groups = [keys] if BATCHED else [[key] for key in keys]
    outputs = {}
    for group in groups:
        rows = generate_rows(
            [contexts[key] for key in group], sample=True, max_new_tokens=generation.max_new_tokens
        )
        for key, new in zip(group, rows, strict=True):
            ended = EOS in new
            if ended:
                new = new[: new.index(EOS) + 1]
            text = tokenizer.decode(new[:-1] if ended else new, skip_special_tokens=False)
            outputs[key] = Generated(new, text, ended)
    if was_training:
        model.train()
    return outputs


rollout_path = os.path.join(OUT, "grpo_rollouts.jsonl")
report["groups"] = []


def on_rollouts(step: int, rollouts) -> None:
    report["groups"].append(
        {
            "step": step,
            "task_id": rollouts[0].task_id,
            "rewards": [r.reward for r in rollouts],
            "terminal_reasons": [r.terminal_reason for r in rollouts],
            "sampled_tokens": [sum(r.env_mask) for r in rollouts],
            "masked_tokens": [len(r.env_mask) - sum(r.env_mask) for r in rollouts],
            "elapsed_seconds": round(time.time() - STARTED, 1),
        }
    )
    with open(rollout_path, "a") as stream:
        for rollout in rollouts:
            stream.write(json.dumps({"step": step, **rollout.trace()}, sort_keys=True) + "\n")
    log(json.dumps(report["groups"][-1]))
    save_report()


class Budget(TrainerCallback):
    """Save adapter checkpoints and stop cleanly before the wall-clock budget runs out."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % RUN["save_every"] == 0:
            target = os.path.join(OUT, f"checkpoint-{state.global_step}")
            model.save_pretrained(target, selected_adapters=["default"])
        if time.time() > STOP_AT:
            report["stopped_by_time_budget_at_step"] = state.global_step
            control.should_training_stop = True
        return control


with open(os.path.join(DATA, "train.jsonl")) as stream:
    train_tasks = [BugTask.from_dict(json.loads(line)) for line in stream.read().splitlines()]
rows, by_prompt = training_prompts(train_tasks)
rollout_func = make_rollout_func(by_prompt, generate, render, client, generation, on_rollouts)
args = GRPOConfig(
    output_dir=os.path.join(OUT, "trainer"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=NUM_GENERATIONS,
    num_generations=NUM_GENERATIONS,
    max_steps=RUN["max_steps"],
    learning_rate=RUN["learning_rate"],
    beta=BETA,
    seed=SEED,
    fp16=True,
    gradient_checkpointing=True,
    disable_dropout=True,
    temperature=generation.temperature,
    max_completion_length=generation.max_context_tokens,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[episode_reward],
    args=args,
    train_dataset=Dataset.from_list(rows),
    processing_class=tokenizer,
    rollout_func=rollout_func,
    callbacks=[Budget()],
)

# Logit check (after TRL patched the forward): with adapters disabled, the model's
# forward and TRL's fused path must reproduce the original log-probs.
model.eval()
probe_labels = LOGIT_PROBE.clone()
probe_labels[:, :-PROBE_TOKENS] = -100
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    with model.disable_adapter():
        fused = model(input_ids=LOGIT_PROBE, labels=probe_labels, fused_lm_head=True)
        folded_forward = forward_logprobs(model)
    start_fused = model(input_ids=LOGIT_PROBE, labels=probe_labels, fused_lm_head=True)
report["logit_check"] = {
    "probe_tokens": PROBE_TOKENS,
    "max_abs_diff_forward": float((folded_forward - original_logprobs).abs().max()),
    "max_abs_diff_fused": float(
        (fused.log_probs[0, -PROBE_TOKENS:].float() - original_logprobs).abs().max()
    ),
    "base_fused_mean_entropy": float(fused.entropy[0, -PROBE_TOKENS:].float().mean()),
    "start_fused_mean_entropy": float(start_fused.entropy[0, -PROBE_TOKENS:].float().mean()),
    "start_fused_mean_logprob": float(start_fused.log_probs[0, -PROBE_TOKENS:].float().mean()),
    "head_untied": head.weight.data_ptr() != base.get_input_embeddings().weight.data_ptr(),
}
log(json.dumps(report["logit_check"]))
save_report()
check = report["logit_check"]
if (
    max(check["max_abs_diff_forward"], check["max_abs_diff_fused"]) > 0.1
    or not check["head_untied"]
):
    raise SystemExit(f"Folded logit scale does not reproduce Dory's log-probs: {check}")
report["trainer"] = {
    "loss_type": args.loss_type,
    "scale_rewards": args.scale_rewards,
    "num_iterations": args.num_iterations,
    "steps_per_generation": args.steps_per_generation,
    "disable_dropout": args.disable_dropout,
    "adapters": sorted(model.peft_config),
}
torch.cuda.reset_peak_memory_stats(0)
model.train()
trainer.train()
report["log_history"] = trainer.state.log_history
report["global_step"] = trainer.state.global_step
report["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)

params = dict(model.named_parameters())
report["parameter_checks"] = {
    "trained_tensors_changed": sum(
        not torch.equal(initial[n], params[n].detach()) for n in initial
    ),
    "trained_tensors": len(initial),
    "ref_adapter_equals_initial": all(
        torch.equal(
            initial[n].to(params[n.replace(".default.", ".ref.")].dtype),
            params[n.replace(".default.", ".ref.")].detach(),
        )
        for n in initial
        if ".default." in n
    ),
    "base_sample_unchanged": all(
        model.get_parameter(n).detach().float().sum().item() == value
        for n, value in base_before.items()
    ),
}
model.save_pretrained(os.path.join(OUT, "adapter"), selected_adapters=["default"])
with open(os.path.join(OUT, "adapter", "adapter_model.safetensors"), "rb") as stream:
    report["adapter_sha256"] = hashlib.sha256(stream.read()).hexdigest()
report["wall_seconds"] = round(time.time() - STARTED, 1)
save_report()
log(json.dumps(report["parameter_checks"]))
checks = report["parameter_checks"]
if not (
    checks["trained_tensors_changed"] > 0
    and checks["ref_adapter_equals_initial"]
    and checks["base_sample_unchanged"]
):
    raise SystemExit(f"Parameter checks failed: {checks}")
