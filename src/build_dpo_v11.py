"""
build_dpo_v11.py
v1.0 DPO 쌍에서 '형식 단서'를 제거한 v1.1 선호 데이터를 만든다.

v1.0 문제: 판례 변경 쌍의 chosen에만 "전원합의체", "(이 판결로 종전 … 변경되었다)",
"종전 판례의 관점이다" 같은 문구가 있었고, rejected에는 없었다. DPO 모델은 판례 내용 대신
이 문구로 선호를 구별했고, 평가셋에서 가공 전원합의체 판결을 인용했다(docs/error_analysis.md 6절).

v1.1 규칙:
1. precedent_conflict 쌍만 사용한다. chosen과 rejected 모두 "대법원 YYYY. M. D. 선고 사건번호 판결: 요지"
   형식이 되도록 chosen에서 '전원합의체', 판례 변경 괄호, 종전/변경을 언급하는 문장을 지운다.
   차이는 적용한 판례(사건번호·요지)와 결론에만 남는다.
2. rule_precedent_exception 쌍은 제외한다. chosen만 판례를 인용하고 rejected는 조문만 인용하는 구조라
   '판례를 인용하면 선호'라는 형식 단서를 없앨 수 없다.
3. 검증용 쌍: 시드 판례 단위로 2건(7쌍)을 학습에서 빼서, 학습에 쓰지 않은 쌍의 선호 정확도로 과적합을 본다.
4. 변환 후 chosen/rejected 어디에도 단서 문구가 남지 않았는지 검사한다(남으면 실패).

출력: data/legal_dpo_dataset_v11.jsonl (split: train/val), data/dpo_v11_report.json
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/legal_dpo_dataset.jsonl"
OUT = ROOT / "data/legal_dpo_dataset_v11.jsonl"
REPORT = ROOT / "data/dpo_v11_report.json"

VAL_SEEDS = {"2012후4162", "2014후768"}          # 검증용으로 빼는 시드 판례
CUE_RE = re.compile(r"전원합의체|종전|변경|현재 판례|이전의")
NEG_PREMISE = "지문은 위 판례와 맞지 않는다."


def split_sections(text):
    m = re.match(r"\[대전제\](.*?)\[소전제\](.*?)\[결론\](.*)", text, re.S)
    if not m:
        raise ValueError("삼단논법 형식이 아님: " + text[:80])
    return [s.strip() for s in m.groups()]


def drop_cue_sentences(s):
    # 문장 단위로 나누고 단서 문구가 들어간 문장을 뺀다.
    parts = re.split(r"(?<=[가-힣A-Za-z)][.!?])\s+", s.strip())  # 날짜(2025. 9. 18.)는 나누지 않음
    keep = [p for p in parts if p and not CUE_RE.search(p)]
    return " ".join(keep).strip()


def clean_chosen(text):
    major, minor, concl = split_sections(text)
    major = re.sub(r"\s*\(이 판결로[^)]*\)", "", major)
    major = major.replace(" 전원합의체 판결", " 판결")
    minor = re.sub(r"\([^)]*변경[^)]*\)", "", minor)
    minor = drop_cue_sentences(minor) or NEG_PREMISE
    # 결론: 첫 문장(O/X 또는 판단)은 유지하고, 단서 문장만 제거
    concl = drop_cue_sentences(concl)
    if not concl:
        raise ValueError("결론이 비었음")
    return f"[대전제] {major}\n[소전제] {minor}\n[결론] {concl}"


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    out, dropped = [], []
    for r in rows:
        if r["category"] != "precedent_conflict":
            dropped.append(r["id"])
            continue
        c = clean_chosen(r["chosen"])
        rej = r["rejected"]
        for name, t in (("chosen", c), ("rejected", rej)):
            if CUE_RE.search(t):
                raise SystemExit(f"[FAIL] {r['id']} {name}에 단서 문구 남음: {CUE_RE.findall(t)}")
        if r["case_no"] not in c:
            raise SystemExit(f"[FAIL] {r['id']} chosen에 근거 사건번호 없음")
        if not any(o in rej for o in r["overruled_cases"]):
            raise SystemExit(f"[FAIL] {r['id']} rejected에 종전 판례 사건번호 없음")
        out.append({**r, "id": r["id"].replace("dpo-", "dpo11-"), "chosen": c,
                    "split": "val" if r["case_no"] in VAL_SEEDS else "train",
                    "meta": {**r["meta"], "v11": "format-matched (build_dpo_v11.py)"}})

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lc = sum(len(r["chosen"]) for r in out) / len(out)
    lr = sum(len(r["rejected"]) for r in out) / len(out)
    rep = {"n_pairs": len(out),
           "n_train": sum(r["split"] == "train" for r in out),
           "n_val": sum(r["split"] == "val" for r in out),
           "val_seeds": sorted(VAL_SEEDS),
           "dropped_rule_precedent_exception": dropped,
           "avg_len_chosen": round(lc), "avg_len_rejected": round(lr),
           "cue_words_checked": CUE_RE.pattern}
    REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
