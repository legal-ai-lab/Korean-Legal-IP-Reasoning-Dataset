"""학습·평가에서 공통으로 쓰는 시스템 프롬프트와 정답 파서."""
import re

SYSTEM = ("당신은 대한민국 법령과 대법원 판례에 근거해 답하는 법률 추론 보조자입니다. "
          "현행 법령과 최신 판례를 기준으로 판단하고, 판례가 변경된 경우 변경된 판례를 따릅니다. "
          "답변은 [대전제], [소전제], [결론] 순서의 법률 삼단논법으로 작성합니다.")

_OX = {"O": "O", "X": "X", "○": "O", "×": "X", "o": "O", "x": "X"}


def parse_ox(text: str) -> tuple[str | None, bool]:
    """모델 출력에서 O/X 정답을 추출한다. (정답, 삼단논법 형식 준수 여부)"""
    fmt_ok = all(tag in text for tag in ("[대전제]", "[소전제]", "[결론]"))
    m = re.search(r"\[결론\]\s*[:：]?\s*\**\s*([OX○×])(?![A-Za-z])", text)
    if not m:
        m = re.search(r"(?:정답|결론|답)\s*(?:은|는|:|：)?\s*\**\s*[\"'“]?([OX○×])(?![A-Za-z])", text)
    if m:
        return _OX[m.group(1)], fmt_ok
    head = text[:300]
    if re.search(r"틀리|틀렸|틀립|옳지 않|맞지 않|부정확", head):
        return "X", fmt_ok
    if re.search(r"옳다|옳습니다|맞다|맞습니다|타당하다", head):
        return "O", fmt_ok
    return None, fmt_ok


def build_messages(user: str, assistant: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs
