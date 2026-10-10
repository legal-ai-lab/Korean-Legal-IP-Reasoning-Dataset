"""
generate_sft_dpo.py
LLM으로 SFT(instruction-output) / DPO(prompt-chosen-rejected) / 평가 문항을 1차 생성한다.

- 생성 모델: claude-opus-5-5 (Anthropic API)
- 프롬프트: prompts/system.md, prompts/{sft,dpo,eval}_generation.md
- 생성 계획: data/generation_plan.json (어떤 원천을 어떤 유형으로 몇 문항 만들지)
- 출력: data/generated/{sft,dpo,eval}_raw.jsonl  →  src/build_dataset.py 로 검수·조립

공개 데이터(v1.0.0)의 1차 생성은 동일 모델·동일 프롬프트를 대화형 인터페이스에서 실행해 얻었다.
이 스크립트는 같은 과정을 API로 재현하기 위한 것이다(LLM 특성상 문장은 매번 달라질 수 있음).

사용법:
    export ANTHROPIC_API_KEY=...
    python src/generate_sft_dpo.py --task sft --out data/generated/sft_raw.regen.jsonl
    python src/build_dataset.py   # 검수(사건번호·조문 대조, 누수 검사)
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def code_block(md_path: Path, idx: int = 0) -> str:
    """프롬프트 문서(.md)에서 idx번째 ``` 코드블록을 꺼낸다."""
    blocks = re.findall(r"```\n(.*?)```", md_path.read_text(encoding="utf-8"), flags=re.S)
    return blocks[idx].strip()


STATUTES = {(r["law_name"], r["article_no"]): r for r in load_jsonl(RAW / "statutes.jsonl")}
PRECS = {r["case_no"].split(",")[0].strip(): r for r in load_jsonl(RAW / "precedents.jsonl")}
SEEDS = {r["case_no"]: r for r in load_jsonl(RAW / "dpo_seeds.jsonl")}


def render_source(src: str) -> str:
    """'특허법 제29조 제2항' 또는 사건번호 → 원문 텍스트."""
    m = re.match(r"^(특허법|상표법|민법)\s+(제\d+조(?:의\d+)?)", src)
    if m:
        r = STATUTES[(m.group(1), m.group(2))]
        return f"[조문] {src}\n{r['text']}"
    r = PRECS[src]
    out = f"[판례] {r['case_no']} (선고 {r['decision_date']})\n판결요지: {r['summary']}"
    if src in SEEDS:
        s = SEEDS[src]
        out += f"\n판례 변경 선언: {s['overrule_sentence']}\n변경된 종전 판례: {', '.join(s['overruled_cases'])}"
    return out


def call(client, model, system, user, max_tokens=4000) -> str:
    msg = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def parse_jsonl(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip().strip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sft", "dpo", "eval"], required=True)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--plan", default=str(ROOT / "data/generation_plan.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개 계획만 실행(테스트용)")
    args = ap.parse_args()

    import anthropic  # pip install anthropic
    client = anthropic.Anthropic()
    system = code_block(ROOT / "prompts/system.md")
    plan = json.load(open(args.plan, encoding="utf-8"))[args.task]
    if args.limit:
        plan = plan[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with out.open("w", encoding="utf-8") as f:
        for i, job in enumerate(plan, 1):
            if args.task == "sft":
                tmpl = code_block(ROOT / "prompts/sft_generation.md")
                source = "\n\n".join(render_source(s) for s in job["sources"])
                user = (tmpl.replace("{CATEGORY}", job["category"]).replace("{N}", str(job["n"]))
                        .replace("{SOURCE}", source))
            elif args.task == "dpo":
                idx = 0 if job["category"] == "precedent_conflict" else 1
                tmpl = code_block(ROOT / "prompts/dpo_generation.md", idx)
                user = tmpl + "\n\n[원문]\n" + render_source(job["case_no"]) + f"\n\n생성 개수: {job['n']}"
            else:
                tmpl = code_block(ROOT / "prompts/eval_generation.md")
                source = "\n\n".join(render_source(s) for s in job["sources"])
                user = tmpl.replace("{N}", str(job["n"])).replace("{SOURCE}", source)

            rows = parse_jsonl(call(client, args.model, system, user))
            for j, r in enumerate(rows):
                r["id"] = f"{args.task}-regen-{i:03d}-{j}"
                r.setdefault("category", job["category"])
                r.setdefault("law", job["law"])
                if args.task == "dpo":
                    r.setdefault("case_no", job["case_no"])
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_ok += 1
            print(f"[{i}/{len(plan)}] {job.get('category')} {job.get('sources') or job.get('case_no')} -> {len(rows)}")
    print(f"[OK] {n_ok} rows -> {out}")


if __name__ == "__main__":
    main()
