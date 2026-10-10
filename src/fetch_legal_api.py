"""
fetch_legal_api.py
국가법령정보 공동활용 Open API(law.go.kr DRF)로 특허법·상표법·민법 조문과
대법원 판례를 수집해 JSONL로 저장한다.

법령·판결은 저작권법 제7조(보호받지 못하는 저작물)에 해당하여 원문 공개가 가능하다.

사용법:
    export LAW_OC=발급받은_OC   # open.law.go.kr 신청 시 이메일 ID
    python src/fetch_legal_api.py --max-prec 300 --out data/raw

출력:
    data/raw/statutes.jsonl    # 조문 1개 = 1행
    data/raw/precedents.jsonl  # 판례 1건 = 1행
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

BASE = "http://www.law.go.kr/DRF"
OC = os.environ.get("LAW_OC", "")

LAWS = ["특허법", "상표법", "민법"]

# 판례 본문 검색 키워드(법령별). 필요 시 추가.
PREC_QUERIES = {
    "특허법": ["특허법", "진보성", "실시권", "특허권 침해"],
    "상표법": ["상표법", "상표의 유사", "식별력", "상표권 침해"],
    "민법": ["민법", "계약자유", "강행규정", "부당이득"],
}

# 반드시 포함할 판례: DPO 핵심 사례 + data/case_seeds.json 후보. 사건번호로 직접 조회.
SEED_PATH = Path("data/case_seeds.json")


def load_pinned() -> list[str]:
    cases = {"2021두59908"}
    if SEED_PATH.exists():
        seeds = json.loads(SEED_PATH.read_text(encoding="utf-8"))
        for group in seeds.values():
            for k, v in group.items():
                if not k.startswith("_"):
                    cases.update(v)
    return sorted(cases)


PINNED_CASES = load_pinned()

# 판례 변경을 선언하는 정형 문구(전원합의체 판결 말미)
OVERRULE_RE = re.compile(r"배치되는\s*범위\s*(내|안)?에서.{0,30}?변경|저촉되는\s*범위\s*(내|안)?에서.{0,10}?변경|변경하기로\s*한다|종전\s*(판례|법리|대법원)[^.]{0,80}더\s*이상\s*유지될\s*수\s*없")

SUPREME_COURT = "400201"  # 법원종류 코드: 대법원


# ---------------------------------------------------------------- utils
def clean(text) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        text = "\n".join(clean(t) for t in text)
    text = re.sub(r"<br\s*/?>", "\n", str(text))
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def get(endpoint: str, params: dict, retries: int = 3) -> dict:
    params = {"OC": OC, "type": "JSON", **params}
    for i in range(retries):
        try:
            r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if i == retries - 1:
                print(f"[WARN] {endpoint} {params.get('query') or params.get('ID') or params.get('MST')}: {e}")
                return {}
            time.sleep(2 ** i)
    return {}


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {len(rows)} rows -> {path}")


# ---------------------------------------------------------------- 법령
def find_law_mst(name: str) -> dict | None:
    """현행 법령 중 법령명이 정확히 일치하는 항목의 MST(법령일련번호)를 찾는다."""
    data = get("lawSearch.do", {"target": "law", "query": name, "display": 20})
    for item in as_list(data.get("LawSearch", {}).get("law")):
        if item.get("법령명한글") == name:
            return item
    return None


def flatten_article(jo: dict) -> str:
    """조문내용 + 항 + 호 + 목을 하나의 텍스트로 합친다."""
    parts = [clean(jo.get("조문내용"))]
    for hang in as_list(jo.get("항")):
        if isinstance(hang, dict):
            parts.append(clean(hang.get("항내용")))
            for ho in as_list(hang.get("호")):
                if isinstance(ho, dict):
                    parts.append(clean(ho.get("호내용")))
                    for mok in as_list(ho.get("목")):
                        if isinstance(mok, dict):
                            parts.append(clean(mok.get("목내용")))
    return "\n".join(p for p in parts if p)


def fetch_statute(name: str) -> list[dict]:
    meta = find_law_mst(name)
    if not meta:
        print(f"[WARN] 법령 검색 실패: {name}")
        return []
    mst = meta.get("법령일련번호")
    data = get("lawService.do", {"target": "law", "MST": mst})
    law = data.get("법령", {})
    info = law.get("기본정보", {})
    rows = []
    for jo in as_list(law.get("조문", {}).get("조문단위")):
        if jo.get("조문여부") != "조문":  # 장·절 제목 제외
            continue
        no = jo.get("조문번호", "")
        sub = jo.get("조문가지번호")
        article_no = f"제{no}조" + (f"의{sub}" if sub else "")
        rows.append({
            "law_name": name,
            "law_mst": mst,
            "enforce_date": info.get("시행일자") or meta.get("시행일자"),
            "article_no": article_no,
            "article_title": clean(jo.get("조문제목")),
            "text": flatten_article(jo),
            "source": f"https://www.law.go.kr/법령/{name}/{article_no}",
        })
    print(f"[OK] {name}: {len(rows)}개 조문")
    return rows


# ---------------------------------------------------------------- 판례
def search_prec(query: str = "", nb: str = "", max_n: int = 100) -> list[dict]:
    items, page = [], 1
    while len(items) < max_n:
        params = {"target": "prec", "display": 100, "page": page,
                  "org": SUPREME_COURT, "sort": "ddes"}
        if nb:
            params["nb"] = nb
        else:
            params.update({"query": query, "search": 2})  # 2 = 본문 검색
        res = as_list(get("lawSearch.do", params).get("PrecSearch", {}).get("prec"))
        if not res:
            break
        items.extend(res)
        if nb or len(res) < 100:
            break
        page += 1
        time.sleep(0.3)
    return items[:max_n]


def fetch_prec_detail(prec_id: str) -> dict | None:
    d = get("lawService.do", {"target": "prec", "ID": prec_id}).get("PrecService")
    if not d:
        return None
    return {
        "prec_id": prec_id,
        "case_no": clean(d.get("사건번호")),
        "case_name": clean(d.get("사건명")),
        "court": clean(d.get("법원명")),
        "decision_date": clean(d.get("선고일자")),
        "decision_type": clean(d.get("판결유형")),
        "en_banc": "전원합의체" in clean(d.get("판결유형")) + clean(d.get("판례내용"))[:500],
        "holdings": clean(d.get("판시사항")),
        "summary": clean(d.get("판결요지")),
        "ref_articles": clean(d.get("참조조문")),
        "ref_precedents": clean(d.get("참조판례")),
        "full_text": clean(d.get("판례내용")),
        "overrules": bool(OVERRULE_RE.search(clean(d.get("판례내용")))),
        "source": f"https://www.law.go.kr/precInfoP.do?precSeq={prec_id}",
    }


def fetch_precedents(max_prec: int) -> list[dict]:
    ids: dict[str, str] = {}  # prec_id -> topic
    for case_no in PINNED_CASES:
        for it in search_prec(nb=case_no):
            ids[it["판례일련번호"]] = "pinned"
    per_q = max(1, max_prec // sum(len(v) for v in PREC_QUERIES.values()))
    for law, qs in PREC_QUERIES.items():
        for q in qs:
            for it in search_prec(query=q, max_n=per_q):
                ids.setdefault(it["판례일련번호"], law)

    rows, seen = [], set()
    for i, (pid, topic) in enumerate(ids.items(), 1):
        d = fetch_prec_detail(pid)
        if d and d["summary"] and d["case_no"] not in seen:  # 요지 없는 건·중복 제외
            d["topic"] = topic
            seen.add(d["case_no"])
            rows.append(d)
        if i % 20 == 0:
            print(f"  판례 상세 {i}/{len(ids)}")
        time.sleep(0.3)

    missing = [c for c in PINNED_CASES if c not in {r["case_no"] for r in rows}]
    if missing:
        print(f"[WARN] 고정 판례 미수집: {missing} — 사건번호/API 반영 여부 확인 필요")
    print(f"[OK] 판례 {len(rows)}건 (전원합의체 {sum(r['en_banc'] for r in rows)}건, "
          f"판례변경 선언 {sum(r['overrules'] for r in rows)}건)")
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--max-prec", type=int, default=300)
    ap.add_argument("--skip-prec", action="store_true")
    args = ap.parse_args()

    if not OC:
        raise SystemExit("환경변수 LAW_OC를 설정하세요 (open.law.go.kr 신청 ID).")

    out = Path(args.out)
    statutes = [r for name in LAWS for r in fetch_statute(name)]
    write_jsonl(statutes, out / "statutes.jsonl")

    if not args.skip_prec:
        write_jsonl(fetch_precedents(args.max_prec), out / "precedents.jsonl")


if __name__ == "__main__":
    main()
