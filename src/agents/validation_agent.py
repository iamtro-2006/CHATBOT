import json
import re
import unicodedata
from typing import Any, Dict


_VALIDATION_PROMPT = """You are an intent classifier for an English-learning app menu.
Task: Determine whether the user wants to start learning now.

Rules:
- Return JSON only, no markdown.
- JSON schema:
  {{
    "should_start": boolean,
    "confidence": number,
    "reason": string
  }}
- should_start=true for clear start intent:
  examples: "start", "begin", "let's start", "I want to learn now", "study now", "learn now", "muon hoc", "hoc ngay".
- should_start=false for negative or postponing intent:
  examples: "not now", "later", "don't start", "chua hoc", "khong hoc".
- If unclear, return should_start=false with low confidence.

User message:
{message}
"""

_NEGATIVE_PATTERNS = [
    r"\bdon'?t\b",
    r"\bdo not\b",
    r"\bnot now\b",
    r"\bnot yet\b",
    r"\blater\b",
]

_START_PATTERNS = [
    r"\bstart\b",
    r"\bbegin\b",
    r"\blet'?s start\b",
    r"\bi want to learn\b",
    r"\bi want to study\b",
    r"\bstudy now\b",
    r"\blearn now\b",
]

_CLASSIFIER_LLM = None


def _to_ascii_lower(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", without_marks).strip().lower()


def _regex_classify(normalized_message: str) -> Dict[str, str | bool | float]:
    if not normalized_message:
        return {
            "should_start": False,
            "confidence": 0.0,
            "reason": "empty_message",
            "normalized_message": "",
        }

    for pattern in _NEGATIVE_PATTERNS:
        if re.search(pattern, normalized_message):
            return {
                "should_start": False,
                "confidence": 0.05,
                "reason": "negative_intent_detected_regex",
                "normalized_message": normalized_message,
            }

    for pattern in _START_PATTERNS:
        if re.search(pattern, normalized_message):
            return {
                "should_start": True,
                "confidence": 0.95,
                "reason": "start_intent_detected_regex",
                "normalized_message": normalized_message,
            }

    return {
        "should_start": False,
        "confidence": 0.15,
        "reason": "no_start_intent_regex",
        "normalized_message": normalized_message,
    }


def _parse_json_from_text(raw_text: str) -> Dict[str, Any]:
    raw = (raw_text or "").strip()
    if not raw:
        raise ValueError("empty model response")
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object in model response")
    return json.loads(raw[start : end + 1])


def _get_classifier_llm():
    global _CLASSIFIER_LLM
    if _CLASSIFIER_LLM is not None:
        return _CLASSIFIER_LLM

    # Lazy import to keep this module test-friendly.
    from utils.load_model import load_chat_model

    _CLASSIFIER_LLM = load_chat_model(
        "google_vertexai/gemini-2.5-flash",
        tags=["validation"],
        temperature=0.0,
    )
    return _CLASSIFIER_LLM


def _prompt_classify(normalized_message: str, llm: Any = None) -> Dict[str, Any]:
    model = llm or _get_classifier_llm()
    prompt = _VALIDATION_PROMPT.format(message=normalized_message)
    response = model.invoke(prompt)
    content = getattr(response, "content", str(response))
    payload = _parse_json_from_text(str(content))
    should_start = bool(payload.get("should_start", False))
    confidence = float(payload.get("confidence", 0.2))
    reason = str(payload.get("reason") or "llm_intent_classification")
    return {
        "should_start": should_start,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": reason,
    }


def validate_start_intent(
    message: str,
    *,
    use_prompt: bool = True,
    llm: Any = None,
) -> Dict[str, str | bool | float]:
    normalized_message = _to_ascii_lower(message or "")
    if not normalized_message:
        return {
            "should_start": False,
            "confidence": 0.0,
            "reason": "empty_message",
            "normalized_message": "",
        }

    if use_prompt:
        try:
            llm_result = _prompt_classify(normalized_message, llm=llm)
            return {
                "should_start": bool(llm_result["should_start"]),
                "confidence": float(llm_result["confidence"]),
                "reason": str(llm_result["reason"]),
                "normalized_message": normalized_message,
            }
        except Exception:
            # Fallback keeps user flow alive if model call fails.
            pass

    return _regex_classify(normalized_message)
