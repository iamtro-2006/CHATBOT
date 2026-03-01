# agents/evaluation_agent.py
import json
from datetime import datetime
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage

from utils.load_model import load_chat_model
from memory.config import MemoryStoreMongo


evaluation_model = load_chat_model(
    "google_vertexai/gemini-2.5-flash",
    tags=["evaluation_agent"],
    temperature=0.0,
)


def _conversation_transcript(chat_history: List[AnyMessage]) -> str:
    parts: List[str] = []
    for m in chat_history:
        content = ("" if m.content is None else str(m.content)).strip()
        if not content:
            continue
        role = "User" if m.__class__.__name__ == "HumanMessage" else "Assistant"
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(s[start:end + 1])
    except Exception:
        return {}


def _score_conversation_with_rubric(conversation_text: str) -> Dict[str, Any]:
    if not conversation_text:
        return {"overall_score": None, "rubric_scores": {}, "feedback": "No conversation to score."}
    sys_msg = SystemMessage(content=(
    "You are a strict but helpful English tutor grading a learner's conversation.\n"
    "You MUST use ONLY the provided conversation and target_vocab list. Do NOT invent details.\n"
    "Return STRICT JSON ONLY. No extra text.\n"
    "\n"
    "Scoring rubric: 4 categories, each 0-5 (integer).\n"
    "Define scores as follows:\n"
    "\n"
    "1) task_completion (0-5): did the learner answer the prompts and satisfy requirements?\n"
    "5 = Fully answers ALL prompts, covers all required parts, enough detail.\n"
    "4 = Answers prompts but misses a minor detail or is slightly short.\n"
    "3 = Partially answers; misses at least one required part OR too vague.\n"
    "2 = Mostly off-topic OR very incomplete.\n"
    "1 = Barely attempts; one-liners; does not follow prompt.\n"
    "0 = No meaningful response.\n"
    "\n"
    "2) vocab_usage (0-5): correct and natural use of target vocabulary.\n"
    "5 = Uses target vocab correctly and naturally (or correct inflections), in meaningful sentences.\n"
    "4 = Uses target vocab correctly but slightly awkward/repetitive.\n"
    "3 = Uses target vocab but with minor meaning/collocation issues.\n"
    "2 = Attempts target vocab but mostly incorrect usage OR only lists words.\n"
    "1 = Rare/incorrect usage; does not demonstrate understanding.\n"
    "0 = Does not use target vocab at all when required.\n"
    "\n"
    "3) grammar_accuracy (0-5): grammar and sentence structure.\n"
    "5 = Mostly accurate; errors are rare and do not affect meaning.\n"
    "4 = Some errors (tense, articles, prepositions) but meaning is clear.\n"
    "3 = Frequent errors; sometimes unclear; limited sentence variety.\n"
    "2 = Many errors; often unclear; fragments/run-ons common.\n"
    "1 = Mostly incorrect grammar; hard to understand.\n"
    "0 = Not English / unintelligible.\n"
    "\n"
    "4) fluency_coherence (0-5): clarity, flow, and organization.\n"
    "5 = Clear, well-structured; ideas connect; good flow.\n"
    "4 = Generally clear; minor awkwardness; mostly coherent.\n"
    "3 = Understandable but choppy; weak linking; some repetition.\n"
    "2 = Disorganized; difficult to follow; many pauses/restarts.\n"
    "1 = Very hard to follow.\n"
    "0 = No coherent output.\n"
    "\n"
    "Penalties / constraints:\n"
    "- If the learner is off-topic for most of the conversation, task_completion MUST be <= 2.\n"
    "- If total learner output is very short (e.g., < 30 words across all learner turns), task_completion MUST be <= 2 and fluency_coherence MUST be <= 2.\n"
    "- If target vocab is required and not used, vocab_usage MUST be 0.\n"
    "\n"
    "Compute overall_score:\n"
    "overall_score = (task_completion + vocab_usage + grammar_accuracy + fluency_coherence) * 5\n"
    "Range 0-100.\n"
    "\n"
    "Feedback requirements:\n"
    "- Vietnamese only.\n"
    "- EXACTLY 3 short sentences, each under 20 words.\n"
    "- MUST follow this exact prefix format:\n"
    "  1) 'Điểm mạnh: ...'\n"
    "  2) 'Điểm yếu: ...'\n"
    "  3) 'Cần cải thiện: ...'\n"
    "- No other text.\n"
"\n"
    "Return JSON exactly in this shape:\n"
    "{\n"
    "  \"overall_score\": 0-100,\n"
    "  \"rubric_scores\": {\n"
    "    \"task_completion\": 0-5,\n"
    "    \"vocab_usage\": 0-5,\n"
    "    \"grammar_accuracy\": 0-5,\n"
    "    \"fluency_coherence\": 0-5\n"
    "  },\n"
    "  \"feedback\": \"...\"\n"
    "}\n"
))

    hm = HumanMessage(content=json.dumps({
        "conversation": conversation_text,
        "note": "Score based only on conversation text.",
    }, ensure_ascii=True))
    try:
        msg = evaluation_model.invoke([sys_msg, hm])
        data = _parse_json_from_text((getattr(msg, "content", "") or "").strip())
        rubric = data.get("rubric_scores") or {}
        overall = data.get("overall_score")
        feedback = (data.get("feedback") or "").strip()
        if isinstance(rubric, dict):
            total = 0
            for k in ["task_completion", "vocab_usage", "grammar_accuracy", "fluency_coherence"]:
                try:
                    total += int(rubric.get(k, 0))
                except Exception:
                    total += 0
            if overall is None:
                overall = max(0, min(100, total * 5))
        else:
            rubric = {}
        if feedback == "":
            feedback = "Cham diem xong. Ban xem lai de tien bo nhe."
        return {"overall_score": overall, "rubric_scores": rubric, "feedback": feedback}
    except Exception:
        return {"overall_score": None, "rubric_scores": {}, "feedback": "Khong the cham diem luc nay."}


def _build_user_feedback(score_pack: Dict[str, Any]) -> str:
    overall = score_pack.get("overall_score")
    rubric = score_pack.get("rubric_scores") or {}
    feedback = (score_pack.get("feedback") or "").strip()

    # Pick weakest areas to mention
    weakness = []
    if isinstance(rubric, dict) and rubric:
        items = []
        for k, v in rubric.items():
            try:
                items.append((k, int(v)))
            except Exception:
                continue
        items.sort(key=lambda x: x[1])
        weakness = [k for k, _ in items[:2]]

    def _name(k: str) -> str:
        return {
            "task_completion": "Hoàn thành đúng yêu cầu",
            "vocab_usage": "Từ vựng",
            "grammar_accuracy": "Ngữ pháp",
            "fluency_coherence": "Độ trôi chảy và mạch lạc",
        }.get(k, k)

    weak_text = ""
    if weakness:
        weak_text = "Can cai thien: " + ", ".join(_name(k) for k in weakness) + ". "

    score_text = f"Diem tong: {overall}/100. " if overall is not None else "Diem tong: chua tinh duoc. "
    cheer = "Bạn làm tốt lắm, cứ tiếp tục phát huy vào ngày sau nhé. "
    tail = feedback + " " if feedback else ""
    return score_text + weak_text + tail + cheer


def evaluate_and_store_session(
    memory_store: MemoryStoreMongo,
    user_id: str,
    thread_id: str,
    chat_history: List[AnyMessage],
) -> Dict[str, Any]:
    if not user_id or not thread_id:
        return {"ok": False, "message": "Khong the cham diem luc nay."}

    thread_blob = memory_store.load_thread(user_id, thread_id) or {}
    speech_progress = thread_blob.get("speech_progress") or {}
    if not speech_progress.get("done"):
        return {"ok": False, "message": ""}

    day_index = int(thread_blob.get("current_day", 1))
    plan_for_day = thread_blob.get(f"last_plan_day_{day_index}") or thread_blob.get("last_plan")
    attempts_by_day = dict(thread_blob.get("session_attempts_by_day") or {})
    last_logged = thread_blob.get("last_session_logged") or {}
    current_attempts = int(attempts_by_day.get(str(day_index), 0))
    already_logged = (
        last_logged.get("thread_id") == thread_id
        and int(last_logged.get("day_index") or 0) == day_index
        and int(last_logged.get("attempt") or 0) == current_attempts
    )
    if already_logged:
        return {"ok": False, "message": ""}

    new_attempt = current_attempts + 1
    transcript = _conversation_transcript(chat_history)
    score_pack = _score_conversation_with_rubric(transcript)
    record_date = datetime.utcnow().date().isoformat()
    record = {
        "thread_id": thread_id,
        "day_index": day_index,
        "date": record_date,
        "attempt": new_attempt,
        "conversation": transcript,
        "plan": plan_for_day,
        "overall_score": score_pack.get("overall_score"),
        "rubric_scores": score_pack.get("rubric_scores"),
        "feedback": score_pack.get("feedback"),
    }
    memory_store.append_session_record(user_id, record)
    attempts_by_day[str(day_index)] = new_attempt
    memory_store.update_thread_fields(user_id, thread_id, {
        "session_attempts_by_day": attempts_by_day,
        "last_session_logged": {
            "thread_id": thread_id,
            "day_index": day_index,
            "attempt": new_attempt,
            "date": record_date,
        },
    })
    user_msg = _build_user_feedback(score_pack)
    return {"ok": True, "message": user_msg, "score": score_pack}
