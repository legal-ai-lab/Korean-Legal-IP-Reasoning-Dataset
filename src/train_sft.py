"""
train_sft.py
Qwen2.5-1.5B-Instruct에 LoRA SFT를 적용한다. (Colab 무료 T4 16GB 기준)

- 입력: data/legal_sft_dataset.jsonl (instruction → output, 법률 삼단논법 CoT)
- 손실: assistant 응답 토큰에만 계산(프롬프트 토큰은 -100으로 마스킹)
- 출력: outputs/qwen2.5-1.5b-legal-lora (LoRA 어댑터), results/train_log.json

사용법:
    python src/train_sft.py --epochs 3
"""
import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)

from prompting import build_messages

ROOT = Path(__file__).resolve().parents[1]


def load_model(name: str):
    kw = dict(device_map="auto")
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, **kw)
    except TypeError:  # 구버전 transformers
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, **kw)


def encode(tok, row, max_len):
    prompt_ids = tok.apply_chat_template(build_messages(row["instruction"]), add_generation_prompt=True, tokenize=True)
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]
    answer_ids = tok(row["output"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            k = n - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_id] * k)
            out["attention_mask"].append(b["attention_mask"] + [0] * k)
            out["labels"].append(b["labels"] + [-100] * k)
        return {k: torch.tensor(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default=str(ROOT / "data/legal_sft_dataset.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "outputs/qwen2.5-1.5b-legal-lora"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    random.Random(args.seed).shuffle(rows)
    ds = [encode(tok, r, args.max_len) for r in rows]
    lens = [len(d["input_ids"]) for d in ds]
    print(f"[data] {len(ds)} samples, tokens mean={sum(lens)/len(lens):.0f} max={max(lens)}")

    model = load_model(args.model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():  # fp16 AMP에서 LoRA 가중치는 fp32로 유지
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(ROOT / "outputs/ckpt"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=2,
        logging_steps=2, save_strategy="no", fp16=True, report_to="none",
        remove_unused_columns=False, seed=args.seed)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=Collator(tok.pad_token_id))
    trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (ROOT / "results").mkdir(exist_ok=True)
    log = {"config": vars(args), "n_samples": len(ds),
           "loss_history": [h for h in trainer.state.log_history if "loss" in h]}
    (ROOT / "results/train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] adapter -> {args.out}")


if __name__ == "__main__":
    main()
