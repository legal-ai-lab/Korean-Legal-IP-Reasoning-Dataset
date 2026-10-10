"""
evaluate.py
평가셋(data/eval_set.jsonl, 학습 미사용 60문항)으로 SFT 전후 OX 정답률을 비교한다.

- 정답률: 전체 / 유형별(standard, rule_precedent_exception, precedent_conflict) / 법령별
- 형식 준수율: [대전제]·[소전제]·[결론]을 모두 갖춘 응답 비율
- 파싱 실패(O/X 미추출)는 오답으로 처리

사용법:
    python src/evaluate.py --name base
    python src/evaluate.py --name sft --adapter outputs/qwen2.5-1.5b-legal-lora
    python src/evaluate.py --report base sft        # results/results.md 생성
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from prompting import build_messages, parse_ox

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CATS = ["standard", "rule_precedent_exception", "precedent_conflict"]
CAT_KO = {"standard": "Standard", "rule_precedent_exception": "Rule-Precedent Exception",
          "precedent_conflict": "Precedent Conflict"}


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    preds = []
    for i in range(0, len(rows), args.bs):
        batch = rows[i:i + args.bs]
        texts = [tok.apply_chat_template(build_messages(r["prompt"]), add_generation_prompt=True, tokenize=False)
                 for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for r, o in zip(batch, out):
            gen = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred, fmt = parse_ox(gen)
            preds.append({"id": r["id"], "category": r["category"], "law": r["law"], "answer": r["answer"],
                          "pred": pred, "correct": pred == r["answer"], "format_ok": fmt,
                          "statement": r["statement"], "output": gen})
        print(f"  {min(i + args.bs, len(rows))}/{len(rows)}")

    RES.mkdir(exist_ok=True)
    with open(RES / f"predictions_{args.name}.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    metrics = summarize(preds)
    metrics["model"] = args.model
    metrics["adapter"] = args.adapter
    (RES / f"metrics_{args.name}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


def summarize(preds):
    def acc(ps):
        return {"n": len(ps), "correct": sum(p["correct"] for p in ps),
                "acc": round(100 * sum(p["correct"] for p in ps) / len(ps), 1) if ps else None}
    by_cat, by_law = defaultdict(list), defaultdict(list)
    for p in preds:
        by_cat[p["category"]].append(p)
        by_law[p["law"]].append(p)
    return {
        "overall": acc(preds),
        "by_category": {c: acc(by_cat[c]) for c in CATS},
        "by_law": {k: acc(v) for k, v in by_law.items()},
        "format_compliance": round(100 * sum(p["format_ok"] for p in preds) / len(preds), 1),
        "parse_fail": sum(p["pred"] is None for p in preds),
        "pred_dist": {k: sum(p["pred"] == k for p in preds) for k in ("O", "X")},
    }


def report(names):
    ms = {n: json.loads((RES / f"metrics_{n}.json").read_text(encoding="utf-8")) for n in names}
    head = "| 구분 | n | " + " | ".join(names) + " |\n|---|---:|" + "---:|" * len(names) + "\n"
    lines = []
    first = ms[names[0]]
    lines.append(f"| **전체** | {first['overall']['n']} | " + " | ".join(f"**{ms[n]['overall']['acc']}%**" for n in names) + " |")
    for c in CATS:
        lines.append(f"| {CAT_KO[c]} | {first['by_category'][c]['n']} | " +
                     " | ".join(f"{ms[n]['by_category'][c]['acc']}%" for n in names) + " |")
    for law in first["by_law"]:
        lines.append(f"| {law} | {first['by_law'][law]['n']} | " +
                     " | ".join(f"{ms[n]['by_law'][law]['acc']}%" for n in names) + " |")
    lines.append("| 삼단논법 형식 준수율 | - | " + " | ".join(f"{ms[n]['format_compliance']}%" for n in names) + " |")
    lines.append("| O/X 파싱 실패 | - | " + " | ".join(str(ms[n]["parse_fail"]) for n in names) + " |")
    md = head + "\n".join(lines) + "\n"
    (RES / "results.md").write_text(md, encoding="utf-8")
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default=str(ROOT / "data/eval_set.jsonl"))
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--report", nargs="+")
    args = ap.parse_args()
    if args.report:
        report(args.report)
    else:
        run(args)


if __name__ == "__main__":
    main()
