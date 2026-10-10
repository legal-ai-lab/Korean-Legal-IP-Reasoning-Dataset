"""
train_dpo.py
SFT 모델 위에 DPO(Direct Preference Optimization)를 LoRA로 학습한다. (Colab 무료 T4 기준)

- 입력: data/legal_dpo_dataset.jsonl (prompt / chosen: 변경 후 판례·판례 해석 / rejected: 종전 판례·조문 문언만 적용)
- 기준 정책(π_ref): Qwen2.5-1.5B-Instruct + SFT LoRA를 병합(merge)한 모델, 학습 중 고정
- 학습 정책(π_θ): 위 병합 모델 + 새 LoRA
- 손실: -log σ( β · [(log π_θ(y_c|x) - log π_ref(y_c|x)) - (log π_θ(y_r|x) - log π_ref(y_r|x))] )
  (trl 버전 차이를 피하기 위해 손실을 직접 구현)
- 출력: outputs/qwen2.5-1.5b-legal-dpo-lora, results/dpo_train_log.json

사용법:
    python src/train_dpo.py --sft_adapter outputs/qwen2.5-1.5b-legal-lora
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from prompting import build_messages

ROOT = Path(__file__).resolve().parents[1]


def load_model(name):
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="auto")
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")


def encode(tok, prompt, response, max_len):
    p = tok.apply_chat_template(build_messages(prompt), add_generation_prompt=True, tokenize=False)
    p_ids = tok(p, add_special_tokens=False)["input_ids"]
    r_ids = tok(response + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + r_ids)[:max_len]
    labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    return ids, labels


def seq_logps(model, seqs, pad_id):
    """seqs: [(ids, labels), ...] → 응답 토큰 log-prob 합 (배치)"""
    dev = next(model.parameters()).device
    n = max(len(a) for a, _ in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    lab = torch.full((len(seqs), n), -100, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, (a, b) in enumerate(seqs):
        ids[i, :len(a)] = torch.tensor(a)
        lab[i, :len(b)] = torch.tensor(b)
        att[i, :len(a)] = 1
    ids, lab, att = ids.to(dev), lab.to(dev), att.to(dev)
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    lab = lab[:, 1:]
    mask = lab != -100
    lp = torch.gather(logits.log_softmax(-1), 2, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sft_adapter", default=str(ROOT / "outputs/qwen2.5-1.5b-legal-lora"))
    ap.add_argument("--data", default=str(ROOT / "data/legal_dpo_dataset.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "outputs/qwen2.5-1.5b-legal-dpo-lora"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    pairs = [(encode(tok, r["prompt"], r["chosen"], args.max_len),
              encode(tok, r["prompt"], r["rejected"], args.max_len), r["category"]) for r in rows]
    print(f"[data] {len(pairs)} pairs")

    # 1) 기준 정책: base + SFT LoRA 병합
    model = load_model(args.model)
    model = PeftModel.from_pretrained(model, args.sft_adapter).merge_and_unload()
    model.config.use_cache = False
    model.eval()

    # 2) 기준 log-prob 사전 계산 (학습 중 고정)
    ref = []
    with torch.no_grad():
        for c, r, _ in pairs:
            lp = seq_logps(model, [c, r], tok.pad_token_id)
            ref.append((lp[0].item(), lp[1].item()))
    print("[ref] reference log-probs computed")

    # 3) 학습 정책: 병합 모델 + 새 LoRA
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    total = (len(pairs) * args.epochs + args.grad_accum - 1) // args.grad_accum
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=2, num_training_steps=total)
    scaler = torch.amp.GradScaler("cuda")

    log, buf = [], []
    order = list(range(len(pairs)))
    step = 0
    for ep in range(args.epochs):
        random.Random(args.seed + ep).shuffle(order)
        for k, i in enumerate(order, 1):
            c, r, cat = pairs[i]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                lp = seq_logps(model, [c, r], tok.pad_token_id)
            margin = args.beta * ((lp[0] - ref[i][0]) - (lp[1] - ref[i][1]))
            loss = -F.logsigmoid(margin) / args.grad_accum
            scaler.scale(loss).backward()
            buf.append((loss.item() * args.grad_accum, margin.item()))
            if k % args.grad_accum == 0 or k == len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                rec = {"step": step, "epoch": round(ep + k / len(order), 3),
                       "loss": round(sum(b[0] for b in buf) / len(buf), 4),
                       "reward_acc": round(sum(b[1] > 0 for b in buf) / len(buf), 3),
                       "reward_margin": round(sum(b[1] for b in buf) / len(buf), 4)}
                log.append(rec)
                print(rec)
                buf = []

    # 4) 학습 후 선호 정확도(학습 쌍 기준, 유형별)
    model.eval()
    acc = {"precedent_conflict": [], "rule_precedent_exception": []}
    with torch.no_grad():
        for i, (c, r, cat) in enumerate(pairs):
            lp = seq_logps(model, [c, r], tok.pad_token_id)
            acc.setdefault(cat, []).append(((lp[0].item() - ref[i][0]) - (lp[1].item() - ref[i][1])) > 0)
    final = {k: round(100 * sum(v) / len(v), 1) for k, v in acc.items() if v}
    print("[pref-acc on train pairs]", final)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/dpo_train_log.json").write_text(json.dumps(
        {"config": vars(args), "n_pairs": len(pairs), "steps": log, "train_pair_pref_acc": final},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] DPO adapter -> {args.out}")


if __name__ == "__main__":
    main()
