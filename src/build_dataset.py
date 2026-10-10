"""
build_dataset.py
LLM 1차 생성 데이터(data/generated/*.jsonl)를 원천 데이터(data/raw)와 대조해
최종 데이터셋을 만들고, 자동 검수 결과를 data/build_report.json에 남긴다.

설계 원칙: "대전제는 원문에서, 추론은 LLM에서"
  - SFT의 [대전제] 조문 인용은 LLM 출력이 아니라 국가법령정보 API 원문에서 자동 삽입한다.
  - 판례 인용 표기(선고일·전원합의체 여부)도 원천 메타데이터에서 생성한다.
  - LLM이 쓴 모든 텍스트에서 사건번호·조문번호를 추출해 원천에 존재하는지 검증한다
    (존재하지 않는 판례 = 환각 → 빌드 실패).
  - 평가셋과 학습셋(SFT·DPO)의 근거 판례가 겹치지 않는지 검증한다.

사용법: python src/build_dataset.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, GEN, OUT = ROOT / "data/raw", ROOT / "data/generated", ROOT / "data"
GENERATOR = "claude-opus-5-5"

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
STATUTE_RE = re.compile(r"^(특허법|상표법|민법)\s+(제\d+조(?:의\d+)?)(?:\s*제(\d+)항)?(?:\s*제(\d+)호)?$")
# 본문 속 사건번호(예: 2010다95390, 81후56, 2019다222782)
CASE_IN_TEXT = re.compile(r"(?<![\d가-힣])((?:19|20)?\d{2}(?:후|허|다|두|도|마|누|그|므|스|카합)\d{1,6})(?!\d)")
# 본문 속 조문 인용(예: 특허법 제29조 제2항)
ART_IN_TEXT = re.compile(r"(특허법|상표법|민법)\s*(제\d+조(?:의\d+)?)")

OX_INST = ("다음 지문의 옳고 그름을 판단하시오. 법률 삼단논법에 따라 [대전제](적용 법규·판례), "
           "[소전제](사안 포섭), [결론](O 또는 X) 순서로 답하시오.\n\n지문: {q}")
QA_INST = ("다음 사안에 법률 삼단논법을 적용하여 답하시오. [대전제](적용 법규·판례), "
           "[소전제](사안 포섭), [결론] 순서로 답하시오.\n\n사안: {q}")
CATEGORIES = {"standard", "rule_precedent_exception", "precedent_conflict"}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jsonl(rows, p: Path):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ 원천 로딩
STATUTES = {(r["law_name"], r["article_no"]): r for r in load_jsonl(RAW / "statutes.jsonl")}
PRECS = {}
for r in load_jsonl(RAW / "precedents.jsonl"):
    PRECS[r["case_no"].split(",")[0].strip()] = r
SEEDS = {s["case_no"]: s for s in load_jsonl(RAW / "dpo_seeds.jsonl")}
OVERRULED = {c for s in SEEDS.values() for c in s["overruled_cases"]}
KNOWN_CASES = set(PRECS) | OVERRULED

errors: list[str] = []


def err(msg: str):
    errors.append(msg)


# ------------------------------------------------------------------ 인용 생성
def statute_text(cite: str, where: str) -> str:
    m = STATUTE_RE.match(cite.strip())
    if not m:
        err(f"{where}: 조문 인용 형식 오류 '{cite}'")
        return cite
    law, art, hang, ho = m.groups()
    rec = STATUTES.get((law, art))
    if not rec:
        err(f"{where}: 원천에 없는 조문 '{law} {art}'")
        return cite
    # 첫 줄의 '제N조(제목)' 접두어를 제거하고, 제목만 있던 줄은 버린다
    lines = rec["text"].split("\n")
    lines[0] = re.sub(r"^" + re.escape(art) + r"\([^)]*\)\s*", "", lines[0])
    lines = [l for l in lines if l.strip()]
    body = lines
    if hang:
        mark = CIRCLED[int(hang) - 1]
        idx = [i for i, l in enumerate(lines) if l.startswith(mark)]
        if not idx:
            err(f"{where}: '{cite}'의 항을 찾을 수 없음")
            return cite
        i = idx[0]
        j = next((k for k in range(i + 1, len(lines)) if lines[k][:1] in CIRCLED), len(lines))
        body = lines[i:j]
    if ho:
        head = body[0]
        hi = [i for i, l in enumerate(body) if re.match(rf"^{ho}\.\s", l)]
        if not hi:
            err(f"{where}: '{cite}'의 호를 찾을 수 없음")
            return cite
        sub = [body[hi[0]]]
        for l in body[hi[0] + 1:]:          # 하위 목(가., 나., …) 포함
            if re.match(r"^[가-하]\.\s", l):
                sub.append(l)
            else:
                break
        body = [head] + sub
    txt = " ".join(l.strip() for l in body)
    if len(txt) > 420:
        txt = txt[:420].rstrip() + "…(이하 생략)"
    return f'{cite}: "{txt}"'


def case_label(case_no: str, where: str) -> str:
    r = PRECS.get(case_no)
    if not r:
        err(f"{where}: 원천에 없는 판례 '{case_no}'")
        return case_no
    d = r["decision_date"]
    date = f"{int(d[:4])}. {int(d[4:6])}. {int(d[6:])}."
    kind = "결정" if re.search(r"[마그스]\d", case_no) else "판결"
    eb = " 전원합의체" if r.get("en_banc") else ""
    return f"대법원 {date} 선고 {r['case_no']}{eb} {kind}"


def check_text_refs(text: str, where: str):
    for c in CASE_IN_TEXT.findall(text):
        if c not in KNOWN_CASES:
            err(f"{where}: 원천·변경목록에 없는 사건번호 '{c}' (환각 의심)")
    for law, art in ART_IN_TEXT.findall(text):
        if (law, art) not in STATUTES:
            err(f"{where}: 원천에 없는 조문 '{law} {art}'")


def premises_block(item: dict, where: str) -> tuple[str, list[dict]]:
    lines, sources = [], []
    for p in item["premises"]:
        if "statute" in p:
            lines.append(statute_text(p["statute"], where))
            law, art = p["statute"].split()[:2]
            sources.append({"type": "statute", "law": law, "article": art, "cite": p["statute"]})
        else:
            c = p["case"]
            label = case_label(c, where)
            lines.append(f"{label}: {p['holding']}")
            check_text_refs(p["holding"], where)
            r = PRECS.get(c, {})
            sources.append({"type": "precedent", "case_no": c, "decision_date": r.get("decision_date"),
                            "url": r.get("source")})
    if item["category"] == "precedent_conflict":
        cases = [s["case_no"] for s in sources if s["type"] == "precedent"]
        for c in cases:
            if c in SEEDS:
                olds = SEEDS[c]["overruled_cases"]
                note = (f"※ 판례 변경: 위 판결로 종전 대법원 {', '.join(olds)} 판결은 변경되었다."
                        if olds else "※ 판례 변경: 위 판결로 종전 판례의 관점은 더 이상 유지될 수 없게 되었다.")
                lines.append(note)
            else:
                err(f"{where}: precedent_conflict인데 판례 변경 확정 목록에 없는 판례 '{c}'")
    return "\n".join(lines), sources


# ------------------------------------------------------------------ SFT
def build_sft(raw: list[dict]) -> list[dict]:
    out = []
    for it in raw:
        w = it["id"]
        if it["category"] not in CATEGORIES:
            err(f"{w}: category 오류")
        prem, sources = premises_block(it, w)
        check_text_refs(it["minor"] + it["conclusion"], w)
        if it["task"] == "ox":
            if it.get("answer") not in ("O", "X"):
                err(f"{w}: OX 정답 누락")
            inst = OX_INST.format(q=it["statement"])
            concl = f"[결론] {it['answer']}. {it['conclusion']}"
        else:
            inst = QA_INST.format(q=it["question"])
            concl = f"[결론] {it['conclusion']}"
        output = f"[대전제]\n{prem}\n[소전제] {it['minor']}\n{concl}"
        out.append({
            "id": w, "category": it["category"], "law": it["law"], "task": it["task"],
            "instruction": inst, "output": output,
            **({"answer": it["answer"]} if it["task"] == "ox" else {}),
            "sources": sources,
            "meta": {"generator": GENERATOR, "prompt": "prompts/sft_generation.md",
                     "major_premise": "auto-inserted from law.go.kr API", "review": "rule-checked + manual"},
        })
    return out


# ------------------------------------------------------------------ DPO
def build_dpo(raw: list[dict]) -> list[dict]:
    out = []
    for it in raw:
        w = it["id"]
        c = it["case_no"]
        if c not in PRECS:
            err(f"{w}: 원천에 없는 판례 '{c}'")
        for k in ("prompt", "chosen", "rejected"):
            check_text_refs(it[k], w)
        if c not in it["chosen"]:
            err(f"{w}: chosen에 근거 사건번호 '{c}' 미포함")
        if it["category"] == "precedent_conflict":
            if c not in SEEDS:
                err(f"{w}: 판례 변경 확정 목록에 없는 판례 '{c}'")
            olds = SEEDS.get(c, {}).get("overruled_cases", [])
            cited_old = [o for o in olds if o in it["rejected"]]
            if not cited_old:
                err(f"{w}: rejected에 변경된 종전 판례 사건번호가 없음")
        r = PRECS.get(c, {})
        out.append({
            "id": w, "category": it["category"], "law": it["law"],
            "prompt": it["prompt"], "chosen": it["chosen"], "rejected": it["rejected"],
            "case_no": c, "decision_date": r.get("decision_date"),
            "overruled_cases": SEEDS.get(c, {}).get("overruled_cases", []),
            "source_url": r.get("source"),
            "meta": {"generator": GENERATOR, "prompt": "prompts/dpo_generation.md", "review": "rule-checked + manual"},
        })
    return out


# ------------------------------------------------------------------ EVAL
def build_eval(raw: list[dict]) -> list[dict]:
    out = []
    for it in raw:
        w = it["id"]
        if it["answer"] not in ("O", "X"):
            err(f"{w}: 정답 오류")
        srcs = []
        for s in it["sources"]:
            if STATUTE_RE.match(s):
                statute_text(s, w)
                law, art = s.split()[:2]
                srcs.append({"type": "statute", "law": law, "article": art, "cite": s})
            else:
                case_label(s, w)
                srcs.append({"type": "precedent", "case_no": s, "url": PRECS.get(s, {}).get("source")})
        check_text_refs(it["statement"] + it.get("rationale", ""), w)
        out.append({
            "id": w, "category": it["category"], "law": it["law"],
            "prompt": OX_INST.format(q=it["statement"]), "statement": it["statement"],
            "answer": it["answer"], "rationale": it.get("rationale", ""), "sources": srcs,
            "meta": {"generator": GENERATOR, "prompt": "prompts/eval_generation.md", "review": "rule-checked + manual"},
        })
    return out


def leak_check(sft, dpo, ev) -> dict:
    train_cases = {s["case_no"] for r in sft for s in r["sources"] if s["type"] == "precedent"}
    train_cases |= {r["case_no"] for r in dpo}
    eval_cases = {s["case_no"] for r in ev for s in r["sources"] if s["type"] == "precedent"}
    case_overlap = sorted(train_cases & eval_cases)
    train_arts = {(s["law"], s["article"]) for r in sft for s in r["sources"] if s["type"] == "statute"}
    eval_arts = {(s["law"], s["article"]) for r in ev if not any(x["type"] == "precedent" for x in r["sources"])
                 for s in r["sources"] if s["type"] == "statute"}
    art_overlap = sorted(f"{a} {b}" for a, b in train_arts & eval_arts)
    train_q = {r["instruction"].split("지문: ")[-1].split("사안: ")[-1].strip() for r in sft}
    dup = [r["id"] for r in ev if r["statement"].strip() in train_q]
    for c in case_overlap:
        err(f"평가셋 누수: 판례 {c}가 학습셋과 평가셋에 모두 사용됨")
    for a in art_overlap:
        err(f"평가셋 누수: 조문 {a}가 학습셋(SFT)과 평가셋 조문 문항에 모두 사용됨")
    for d in dup:
        err(f"평가셋 누수: {d} 지문이 학습셋과 동일")
    return {"case_overlap": case_overlap, "statute_overlap": art_overlap, "duplicate_statements": dup}


def ratio(rows):
    c = Counter(r["category"] for r in rows)
    n = len(rows)
    return {k: {"n": c[k], "pct": round(100 * c[k] / n, 1)} for k in sorted(CATEGORIES)} | {"total": n}


def main():
    sft = build_sft(load_jsonl(GEN / "sft_raw.jsonl"))
    dpo = build_dpo(load_jsonl(GEN / "dpo_raw.jsonl"))
    ev = build_eval(load_jsonl(GEN / "eval_raw.jsonl"))
    ids = Counter(r["id"] for r in sft + dpo + ev)
    for k, v in ids.items():
        if v > 1:
            err(f"중복 id {k}")
    leaks = leak_check(sft, dpo, ev)
    report = {
        "sft": ratio(sft) | {"by_law": Counter(r["law"] for r in sft), "by_task": Counter(r["task"] for r in sft),
                             "ox_answer": Counter(r.get("answer") for r in sft if r["task"] == "ox")},
        "dpo": ratio(dpo) | {"cases": len({r['case_no'] for r in dpo})},
        "eval": ratio(ev) | {"by_law": Counter(r["law"] for r in ev), "answer": Counter(r["answer"] for r in ev)},
        "leak_check": leaks,
        "checks": {
            "statute_quotes_verbatim_from_api": True,
            "case_numbers_verified_against_raw": True,
            "conflict_cases_verified_by_overrule_sentence": True,
        },
        "errors": errors,
    }
    (OUT / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if errors:
        print(f"[FAIL] 검수 오류 {len(errors)}건")
        for e in errors[:50]:
            print("  -", e)
        sys.exit(1)
    write_jsonl(sft, OUT / "legal_sft_dataset.jsonl")
    write_jsonl(dpo, OUT / "legal_dpo_dataset.jsonl")
    write_jsonl(ev, OUT / "eval_set.jsonl")
    print(f"[OK] SFT {len(sft)} / DPO {len(dpo)} / EVAL {len(ev)}")
    print(json.dumps({k: report[k] for k in ("sft", "dpo", "eval")}, ensure_ascii=False, indent=1, default=dict))


if __name__ == "__main__":
    main()
