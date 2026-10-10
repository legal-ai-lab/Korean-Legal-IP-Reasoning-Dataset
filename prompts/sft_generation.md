# SFT 생성 프롬프트

생성 모델: `claude-opus-5-5` · 출력: `data/generated/sft_raw.jsonl`

## 입력
- `{CATEGORY}`: `standard` | `rule_precedent_exception` | `precedent_conflict`
- `{SOURCE}`: 원천 레코드 1건
  - 조문: `data/raw/statutes.jsonl`의 `law_name`, `article_no`, `text`
  - 판례: `data/raw/precedents.jsonl`의 `case_no`, `decision_date`, `summary`(판결요지)
  - 판례 변경: `data/raw/dpo_seeds.jsonl`의 `overrule_sentence`, `overruled_cases`
- `{N}`: 생성 문항 수(1~3)

## 프롬프트
```
아래 원문만을 근거로 {CATEGORY} 유형의 SFT 문항 {N}개를 만드시오.

[원문]
{SOURCE}

[유형 정의]
- standard: 조문 문언 또는 대표 판례 법리를 그대로 묻는 OX/사례형 문항.
- rule_precedent_exception: 조문 문언만 적용하면 오답이 되고, 판례의 해석(제한·확장·예외)을
  알아야 정답이 되는 문항. [소전제]에 "문언상 ~이지만 판례는 ~"의 대비를 반드시 쓴다.
- precedent_conflict: 전원합의체 판결로 변경된 종전 판례의 관점을 지문으로 제시하고,
  현재 판례 기준으로 판단하게 하는 문항. [소전제]에 변경 연도와 변경 사실을 쓴다.

[작성 규칙]
- OX 문항은 정답 O/X 비율이 한쪽으로 치우치지 않게 한다. 오답 지문은 기간·주체·요건·효과 중
  하나만 바꾸어 만든다(함정이 둘 이상이면 안 된다).
- premises에는 근거만 넣는다. 조문은 {"statute": "법령명 제N조 제N항 제N호"},
  판례는 {"case": "사건번호", "holding": "판결요지에 근거한 1~2문장"}.
- holding은 판결요지를 벗어나지 않게 요약하고, 판결요지에 없는 일반론을 추가하지 않는다.
- minor(소전제)는 지문/사안을 대전제에 포섭하는 1~3문장, conclusion은 결론과 그 이유 1문장.

[출력 스키마: JSON Lines]
{"id": "...", "category": "{CATEGORY}", "law": "특허법|상표법|민법", "task": "ox",
 "statement": "...", "answer": "O|X", "premises": [...], "minor": "...", "conclusion": "..."}
{"id": "...", "category": "{CATEGORY}", "law": "...", "task": "qa",
 "question": "사안 + 질문", "premises": [...], "minor": "...", "conclusion": "..."}
```

## 후처리 (`src/build_dataset.py`)
1. `statute` 인용 위치 → 국가법령정보 API 원문을 그대로 삽입(항·호·목 단위)
2. `case` → 선고일·전원합의체 여부를 원천 메타데이터로 표기
3. `precedent_conflict` → 변경된 종전 판례 사건번호를 원문 변경 선언 문장에서 추출해 자동 부기
4. 텍스트 속 사건번호·조문번호 존재 검증, 평가셋 누수 검사
