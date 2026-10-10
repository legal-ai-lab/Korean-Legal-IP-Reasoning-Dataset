# DPO 생성 프롬프트

생성 모델: `claude-opus-5-5` · 출력: `data/generated/dpo_raw.jsonl`

## 입력
- `{SEED}`: `data/raw/dpo_seeds.jsonl` 1건
  (`case_no`, `decision_date`, `summary`, `overrule_sentence`, `overruled_cases`)
- `{OLD_CASE}`: `overruled_cases` 중 1건과 그 선고일(변경 선언 문장에 기재된 것만 사용)

## 프롬프트 A: Precedent Conflict
```
아래 전원합의체 판결과 그 판결로 변경된 종전 판례를 근거로 DPO 쌍 3개를 만드시오.

[현재 판례] {SEED.case_no} ({SEED.decision_date})
판결요지: {SEED.summary}
판례 변경 선언: {SEED.overrule_sentence}

[종전 판례] {OLD_CASE}

[규칙]
- prompt: 직접 질문 1개, 사례형 1개, OX 판단형 1개.
- chosen: 현재 판례를 [대전제]로 한 삼단논법. 판례가 변경되었다는 사실을 포함한다.
- rejected: 종전 판례를 [대전제]로 하여 자신 있게 결론 내리는 답변. "종전", "변경 전" 같은
  문구를 붙이지 않는다(모델이 문구만 보고 구별하는 것을 막기 위함).
- chosen과 rejected는 형식·길이를 비슷하게 맞추고, 차이는 적용한 판례와 결론에만 둔다.
- 사건번호는 위 두 판례 외에는 쓰지 않는다.

[출력 스키마]
{"id": "...", "category": "precedent_conflict", "law": "...", "case_no": "{SEED.case_no}",
 "prompt": "...", "chosen": "...", "rejected": "..."}
```

## 프롬프트 B: Rule-Precedent Exception
```
아래 조문과 판례를 근거로, 조문 문언만 기계적으로 적용한 답(rejected)과
판례 해석을 반영한 답(chosen)으로 구성된 DPO 쌍 1개를 만드시오.

[조문] {STATUTE}
[판례] {CASE.case_no}: {CASE.summary}

[규칙]
- chosen의 [대전제]는 "○○법 제N조의 문언에도 불구하고, 대법원 … 판결: …" 형식.
- rejected의 [대전제]는 조문 문언만 사용하고 판례를 언급하지 않는다.
```

## 자동 검수 (`src/build_dataset.py`)
- chosen에 근거 사건번호 포함 여부
- precedent_conflict의 rejected에 **실제로 변경된 종전 판례 사건번호**가 포함되었는지
- 모든 사건번호가 원천(`precedents.jsonl`) 또는 변경 목록(`overruled_cases`)에 존재하는지
