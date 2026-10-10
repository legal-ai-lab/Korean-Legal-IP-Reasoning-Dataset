"""
extract_dpo_seeds.py
수집된 판례(precedents.jsonl) 중 '판례 변경을 선언한 IP 분야 전원합의체 판결'을 골라
DPO 원천(dpo_seeds.jsonl)을 만든다.

판정 기준(규칙 기반, LLM 미사용)
  1) 전원합의체 판결일 것 (en_banc)
  2) 판결 원문에 판례 변경 선언 문구가 있을 것
     예) "이 판결의 견해에 배치되는 범위에서 이를 변경하기로 한다"
         "이와 저촉되는 범위 내에서 변경하기로 한다"
  3) 특허·상표 사건이거나(사건부호 '후'/'허'), IP 쟁점 민사·행정 사건으로 명시한 것
변경 선언 문장에서 '변경된 종전 판례' 사건번호를 추출해 DPO rejected 근거로 쓴다.

사용법: python src/extract_dpo_seeds.py --raw data/raw
"""
import argparse
import json
import re
from pathlib import Path

# 변경 선언 문장(마지막 문장 단위로 잘라서 사용)
DECLARE_RE = re.compile(
    r"(배치되는|저촉되는)\s*범위\s*(내|안)?에서[^.]{0,20}?변경"
)
CASE_RE = re.compile(
    r"(\d{2,4}\s?[가-힣]{1,2}\s?\d{1,6})\s*(?:,\s*\d+)*\s*(?:\(병합\))?\s*(?:판결|결정)"
)
# '후'/'허' 외에 IP 쟁점을 다룬 전원합의체 판결(민사·행정 사건부호)
IP_EXTRA = {"2010다95390", "2010다103000", "2018다253444", "2021두59908"}


def norm(c: str) -> str:
    return re.sub(r"\s+", "", c)


def declare_sentence(text: str) -> str | None:
    """원문에서 마지막 판례 변경 선언 문장을 반환."""
    last = None
    for m in DECLARE_RE.finditer(text):
        last = m
    if not last:
        return None
    start = text.rfind("다.", 0, last.start())
    start = 0 if start < 0 else start + 2
    end = text.find("다.", last.end())
    end = len(text) if end < 0 else end + 2
    return text[start:end].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    args = ap.parse_args()
    raw = Path(args.raw)
    rows = [json.loads(l) for l in open(raw / "precedents.jsonl", encoding="utf-8")]

    seeds = []
    for r in rows:
        case_no = r["case_no"]
        is_ip = any(t in case_no for t in ("후", "허")) or case_no in IP_EXTRA
        if not (r.get("en_banc") and is_ip):
            continue
        sent = declare_sentence(r["full_text"])
        if not sent:
            continue
        olds = sorted({norm(c) for c in CASE_RE.findall(sent)} - {case_no})
        seeds.append({
            "case_no": case_no,
            "decision_date": r["decision_date"],
            "case_name": r["case_name"],
            "overruled_cases": olds,
            "overrule_sentence": sent,
            "summary": r["summary"],
            "source": r["source"],
        })

    out = raw / "dpo_seeds.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[OK] DPO 원천 {len(seeds)}건 -> {out}")
    for s in seeds:
        print(f"  {s['case_no']} → 변경된 판례 {len(s['overruled_cases'])}건: {', '.join(s['overruled_cases'])}")


if __name__ == "__main__":
    main()
