# agents/supervisor.py
from dotenv import load_dotenv
load_dotenv()
import os, sys, re, time, random, json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AnyMessage
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig

# ===== Setup paths =====
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from utils.extract_text import extract_clean_text
from utils.load_model import load_chat_model
from agents.planner import planner_agent
from agents.speech_agent import speech_agent, speech_step
from pymongo import MongoClient
from memory.config import MemoryStoreMongo
from tools.memory_tools import _simple_concat_summary, init_memory_tools
from agents.evaluation_agent import evaluate_and_store_session

# ===== Mongo memory =====
mongo = MongoClient(os.getenv("MONGO_URI"))
db = mongo[os.getenv("MONGO_DB", "english_coach_db")]

memory_store = MemoryStoreMongo(
    user_profiles_col=db["user_profiles"],
    threads_col=db["threads"],
)

init_memory_tools(memory_store)

# ===== MODEL =====
model = load_chat_model(
    "google_vertexai/gemini-2.5-flash",
    tags=["supervisor"],
    temperature=0.0,
)

# ===== SUPERVISOR PROMPT =====
SUPERVISOR_PROMPT = """
You are the Supervisor of an English listening and speaking coaching chatbot for visually impaired learners.
Your mission: coordinate the conversation, collect the user’s topic choice and the minimum required information, then pass that information to the Planner to generate a lesson plan, store data, and save the generated lesson in the database.

VOICE-FRIENDLY RULES
- Ask only 1 main question per turn. Keep it short, clear, and easy for TTS.
- Provide at most 6 options; prefer 3 options for follow-up questions.
- If you have enough information, do not ask further questions; hand off to the Planner.
- If information is missing, use reasonable defaults instead of asking too many questions.

MINIMUM REQUIREMENTS TO CALL THE PLANNER
- selected_topic (required)
- level (important; if level is already known, do not ask again—use it as default)
- focus (required)
- session_minutes (default: 10)
- scenario (optional)

TOPIC LIST (show exactly 6 options in the first question) — selected_topic:
1. Daily communication
2. Travel
3. Work
4. Study
5. Health
6. Entertainment

COORDINATION GUIDELINES
- First turn: greet and ask the user to choose a topic by number (1–6).
- If the user is vague: suggest 1/2/3 and ask them to choose a number.
- If the user proposes a different topic: accept it and confirm.
- After the topic is chosen: ask two SHORT questions:
  (1) Level — level: beginer / meddium / good
  (2) Goal — focus: listen / speak / both
  If the user does not answer clearly → use defaults.
- Always analyze the user’s message and MAP it to normalized values to extract information:
  Examples:
  - level: "beginner/new/begin" -> beginner; "medium" -> medium; "good/well" -> good
  - focus: "listen" -> listen; "speak" -> speak; "both" -> both
  - selected_topic:
      "daily communication" -> Daily communication
      "travel/trip/vacation" -> Travel
      "work/job/office" -> Work
      "study/school" -> Study
      "health/healthy" -> Health
      "entertainment/movie/music/game" -> Entertainment
- If level is present (non-empty), treat it as known and do not ask again. When creating the payload for the Planner, use level = level.
- The topic is chosen by the user each day; DO NOT reuse the previous selected_topic to generate a plan.
- If level is already known, skip asking level. Still ask for topic and focus.
- On a new day, always create a new plan based on last_rubic_score and the information in user_profile.

WHEN YOU HAVE ENOUGH INFORMATION → HAND OFF TO THE PLANNER
- Do NOT write the lesson plan yourself.
- After the Planner returns JSON: summarize it in 3–5 lines for the user and ask: "Do you want to start now?"
- Do NOT read the entire JSON back to the user.

PLANNER RULES (MANDATORY)
- Only hand off to the Planner after you have enough information.
- To generate the curriculum/lesson plan, you MUST HAND OFF to planner_agent using transfer_to_planner_agent.
- The payload sent to planner_agent must include:
{
  "user_id": "...",
  "thread_id": "...",
  "user_profile_fields": {
    "level": "beginer / meddium / good",
    "focus": "listen / speak / both",
    "session_minutes": 10,
    "accessibility": "voice-friendly"
  },
  "thread_fields": {
    "selected_topic": "...",
    "scenario": "...",
    "start_day": "...",
    "current_day": 1
  }
}
- The Planner will create a lesson appropriate to current_day and the user’s information.
- ABSOLUTELY DO NOT write the lesson plan in the Supervisor.
- ABSOLUTELY DO NOT ask "start?" or "Do you want to start now?" BEFORE the Planner has created the lesson plan.
- Only after the Planner has finished creating the lesson (i.e., you have received the result from planner_agent) may you ask whether the learner wants to start.

CONSTRAINTS
- Do not mention internal agent/tool names in the final user-facing response.
- If the user asks technical/code questions, respond briefly and steer back to the learning goal.
Important: When speaking to the end user, always respond in Vietnamese (voice-friendly), even though these instructions are in English.
"""

# ===== State Schema =====
class SupervisorState(AgentState):
    user_id: str
    thread_id: str
    user_profile_fields: Dict[str, Any]
    thread_fields: Dict[str, Any]
# ===== Create Supervisor =====
supervisor = create_supervisor(
    [planner_agent],
    model=model,
    prompt=SUPERVISOR_PROMPT,
    output_mode="full_history",
    add_handoff_messages=True,
    add_handoff_back_messages=True,
    state_schema=SupervisorState,
    supervisor_name="supervisor",
).compile()

# ===== Utilities =====
TRANSFER_RE = re.compile(r"(^transfer_|handoff)", re.I)
_START_CMDS = {
    "ok",
    "okay",
    "yes",
    "start",
}

def _is_start_cmd(text: str) -> bool:
    return (text or "").strip().lower() in _START_CMDS

def _infer_topic_from_text(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    number_map = {
        "1": "Daily communication",
        "2": "Travel",
        "3": "Work",
        "4": "Study",
        "5": "Health",
        "6": "Entertainment",
    }
    if t in number_map:
        return number_map[t]
    keywords = {
        "communication": "Daily communication",
        "communicate": "Daily communication",
        "daily communication": "Daily communication",
        "travel": "Travel",
        "trip": "Travel",
        "vacation": "Travel",
        "work": "Work",
        "job": "Work",
        "office": "Work",
        "study": "Study",
        "school": "Study",
        "health": "Health",
        "healthy": "Health",
        "entertainment": "Entertainment",
        "movie": "Entertainment",
        "music": "Entertainment",
        "game": "Entertainment",
    }
    for k, v in keywords.items():
        if k in t:
            return v
    return ""

def _infer_topic_from_ai(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    mapping = {
        "communicate": "Daily communication",
        "communication": "Daily communication",
        "daily communication": "Daily communication",
        "travel": "Travel",
        "work": "Work",
        "study": "Study",
        "health": "Health",
        "entertainment": "Entertainment",

    }
    for k, v in mapping.items():
        if k in t:
            return v
    return ""

def _infer_level_from_text(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    if "beginner" in t or "ngÆ°á»i má»›i" in t or "moi" in t:
        return "beginer"
    if "medium" in t or "trung bÃ¬nh" in t:
        return "meddium"
    if "good" in t or "nÃ¢ng cao" in t or "tá»‘t" in t:
        return "good"
    return ""

def _infer_focus_from_text(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    if "both" in t or "cáº£ hai" in t or "ca hai" in t:
        return "both"
    if "listen" in t or "nghe" in t:
        return "listen"
    if "speak" in t or "nÃ³i" in t or "noi" in t:
        return "speak"
    return ""

def _is_empty(val) -> bool:
    """Check if value is empty"""
    if val is None:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False

def _clean_messages_for_llm(msgs: List[AnyMessage]) -> List[AnyMessage]:
    """Clean messages for LLM consumption"""
    cleaned: List[AnyMessage] = []
    for m in msgs:
        if isinstance(m, ToolMessage):
            continue
        content = ("" if m.content is None else str(m.content)).strip()
        if not content:
            continue
        if isinstance(m, HumanMessage):
            cleaned.append(HumanMessage(content=content))
        elif isinstance(m, AIMessage):
            cleaned.append(AIMessage(content=content))
    
    # Ensure last message is HumanMessage
    if not cleaned or not isinstance(cleaned[-1], HumanMessage):
        last_h = next((x for x in reversed(msgs) if isinstance(x, HumanMessage)), None)
        if last_h:
            cleaned.append(HumanMessage(content=str(last_h.content)))
    
    return cleaned

def _normalize_messages_from_sup_out(sup_out):
    """Normalize supervisor output to get messages"""
    if isinstance(sup_out, list):
        return sup_out
    if isinstance(sup_out, dict):
        return (
            sup_out.get("messages")
            or sup_out.get("chat_history")
            or sup_out.get("history")
            or []
        )
    return []

def _last_ai_with_content(msgs: List[AnyMessage]) -> AIMessage | None:
    """Get last AI message with content"""
    def extract_text(content) -> str:
        """Extract text from various content formats."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handle Gemini's list format: [{'type': 'text', 'text': '...'}]
            text_parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif isinstance(item, str):
                    text_parts.append(item)
                elif hasattr(item, 'text'):
                    text_parts.append(item.text)
            return ' '.join(text_parts)
        # Fallback for other types
        return str(content) if content else ""
    
    return next(
        (m for m in reversed(msgs) if isinstance(m, AIMessage) and extract_text(m.content).strip()),
        None
    )

def _last_human_text(msgs: List[AnyMessage]) -> str:
    """Get last user text for scoring."""
    m = next((x for x in reversed(msgs) if isinstance(x, HumanMessage)), None)
    return (m.content or "").strip() if m else ""


def _latest_session_record(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    best_key = (-1, -1)
    for r in records or []:
        try:
            day = int(r.get("day_index", -1))
        except Exception:
            day = -1
        try:
            att = int(r.get("attempt", -1))
        except Exception:
            att = -1
        key = (day, att)
        if key > best_key:
            best_key = key
            best = r
    return best or {}

def _plan_for_current_day(thread_blob: Dict[str, Any]) -> Dict[str, Any]:
    current_day = thread_blob.get("current_day", 1)
    plan = thread_blob.get(f"last_plan_day_{current_day}")
    if plan:
        return plan
    last_plan = thread_blob.get("last_plan") or {}
    meta = (last_plan.get("meta") or {}) if isinstance(last_plan, dict) else {}
    if meta.get("day_index") == current_day:
        return last_plan
    return {}

def _has_valid_plan(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    meta = plan.get("meta") or {}
    lesson = plan.get("lesson") or {}
    if not isinstance(meta, dict) or not isinstance(lesson, dict):
        return False
    if not meta.get("day_index"):
        return False
    return True

def _latest_date(records: List[Dict[str, Any]]) -> str:
    best = ""
    for r in records or []:
        d = str(r.get("date") or "")
        if d and d > best:
            best = d
    return best

def _max_day_index(records: List[Dict[str, Any]]) -> int:
    max_day = 0
    for r in records or []:
        try:
            d = int(r.get("day_index", 0))
        except Exception:
            d = 0
        if d > max_day:
            max_day = d
    return max_day

def evaluation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Score conversation after speech session ends and persist record."""
    if not state.get("use_speech"):
        return state

    user_id = state.get("user_id")
    thread_id = state.get("thread_id")
    if not user_id or not thread_id:
        return state

    result = evaluate_and_store_session(
        memory_store,
        user_id,
        thread_id,
        state.get("chat_history", []),
    )

    if result.get("ok") and result.get("message"):
        new_state = dict(state)
        msgs = list(new_state.get("messages") or [])
        ai = AIMessage(content=result["message"])
        new_state["messages"] = msgs + [ai]
        new_state["chat_history"] = list(new_state.get("chat_history") or []) + [ai]
        new_state["should_exit"] = True
        return new_state

    return state

# ===== Graph Nodes =====
def handle_request(state: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and prepare messages"""
    raw = list(state.get("chat_history") or state.get("messages") or [])
    msgs = _clean_messages_for_llm(raw)
    return {**state, "messages": msgs} if msgs else state

def retrieve_memories(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load user profile and thread data from database"""
    # CRITICAL: Always use the original user_id and thread_id from initial state
    user_id = state.get("user_id")
    thread_id = state.get("thread_id")
    
    if not user_id or not thread_id:
        raise ValueError("user_id and thread_id must be provided in state")

    # Load from Mongo
    user_prof = memory_store.load_user_profile(user_id) or {}
    thread_blob = memory_store.load_thread(user_id, thread_id) or {}

    # Handle incoming overrides
    incoming_level = state.get("level")
    incoming_focus = state.get("focus")
    incoming_minutes = state.get("session_minutes")
    incoming_access = state.get("accessibility")

    if any([incoming_level, incoming_focus, incoming_minutes, incoming_access]):
        memory_store.upsert_user_profile(
            user_id,
            level=incoming_level,
            focus=incoming_focus,
            session_minutes=incoming_minutes,
            accessibility=incoming_access,
        )
        user_prof = memory_store.load_user_profile(user_id) or {}

    records = user_prof.get("session_records") or []
    latest_date = _latest_date(records)
    today = datetime.utcnow().date().isoformat()
    is_new_day = bool(latest_date and latest_date < today)
    next_day = None
    if is_new_day:
        last_day = _max_day_index(records)
        next_day = min(last_day + 1, 4) if last_day else 1
        if int(thread_blob.get("current_day", 1)) != next_day:
            memory_store.update_thread_fields(user_id, thread_id, {"current_day": next_day})
            thread_blob = memory_store.load_thread(user_id, thread_id) or {}

    # Build new state
    new_state = dict(state)

    if not state.get("level") and user_prof.get("level"):
        new_state["level"] = user_prof.get("level")
    
    new_state["learning_profile"] = {
        "level": state.get("level") or user_prof.get("level") or "mới bắt đầu",
        "focus": state.get("focus") or user_prof.get("focus") or "both",
        "session_minutes": int(state.get("session_minutes") or user_prof.get("session_minutes") or 10),
        "accessibility": state.get("accessibility") or user_prof.get("accessibility") or "voice-friendly",
    }
    last_rec = _latest_session_record(user_prof.get("session_records") or [])
    new_state["last_feedback"] = last_rec.get("feedback")
    new_state["last_rubic_score"] = last_rec.get("rubric_scores")
    conv_summary = thread_blob.get("conversation_summary", "")
    last_topic = thread_blob.get("last_topic")

    last_plan = thread_blob.get("last_plan")
    thread_title_val = thread_blob.get("thread_title")

    new_state["retrieved_memory"] = {
        "conversation_summary": conv_summary,
        "last_topic": last_topic,
        "last_plan": last_plan,
    }
    
    new_state["thread_title"] = thread_title_val
    
    thread_state = {
        "selected_topic": thread_blob.get("selected_topic"),
        "scenario": thread_blob.get("scenario"),
        "start_day": thread_blob.get("start_day"),
        "current_day": thread_blob.get("current_day", 1),
        "last_day_result": thread_blob.get("last_day_result"),
        "last_plan": thread_blob.get("last_plan"),
    }
    if is_new_day:
        # force supervisor to ask topic again
        thread_state["selected_topic"] = None

    new_state["info"] = {
        "user_id": user_id,  # Use the same user_id consistently
        "thread_id": thread_id,  # Use the same thread_id consistently
        "learning_profile": new_state["learning_profile"],
        "thread_state": thread_state,
        "conversation_summary": conv_summary,
        "is_new_day": is_new_day,
        "skip_level": bool(user_prof.get("level")),
    }
    new_state["next_day"] = next_day
    plan_ready = _has_valid_plan(_plan_for_current_day(thread_blob) or thread_blob.get("last_plan"))
    # If a plan exists, go straight to speech flow (skip supervisor)
    new_state["use_speech"] = bool(plan_ready)
    new_state["plan_ready"] = plan_ready
    
    # IMPORTANT: Preserve user_id and thread_id in state
    new_state["user_id"] = user_id
    new_state["thread_id"] = thread_id
    
    return new_state

def executor(state: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(state)
    state.setdefault("chat_history", [])
    state.setdefault("messages", [])

    # ===== 1. LOCK user_id & thread_id (TUYỆT ĐỐI) =====
    user_id = state.get("user_id")
    thread_id = state.get("thread_id")
    if not user_id or not thread_id:
        raise ValueError("executor: user_id / thread_id missing")
    # If user confirms start and a current-day plan exists, skip supervisor and go to speech
    latest_user = _last_human_text(state.get("messages") or [])
    inferred_topic = _infer_topic_from_text(latest_user)
    if not inferred_topic:
        prev_ai = _last_ai_with_content(state.get("messages") or [])
        inferred_topic = _infer_topic_from_ai(prev_ai.content if prev_ai else "")
    if inferred_topic:
        thread_blob = memory_store.load_thread(user_id, thread_id) or {}
        if not thread_blob.get("selected_topic"):
            memory_store.update_thread_fields(user_id, thread_id, {"selected_topic": inferred_topic})
    inferred_level = _infer_level_from_text(latest_user)
    if inferred_level:
        memory_store.upsert_user_profile(user_id, level=inferred_level)
    inferred_focus = _infer_focus_from_text(latest_user)
    if inferred_focus:
        memory_store.upsert_user_profile(user_id, focus=inferred_focus)

    if _is_start_cmd(latest_user):
        thread_blob = memory_store.load_thread(user_id, thread_id) or {}
        print(_plan_for_current_day(thread_blob))
        if _has_valid_plan(_plan_for_current_day(thread_blob) or thread_blob.get("last_plan")):
            state["use_speech"] = True
            return state
    if state.get("use_speech"):
        return state

    # If info is complete, transfer directly to planner (skip supervisor)
    thread_blob = memory_store.load_thread(user_id, thread_id) or {}
    user_prof = memory_store.load_user_profile(user_id) or {}
    selected_topic = thread_blob.get("selected_topic") or inferred_topic
    level = user_prof.get("level") or inferred_level
    focus = user_prof.get("focus") or inferred_focus
    if selected_topic and level and focus and not _has_valid_plan(_plan_for_current_day(thread_blob) or thread_blob.get("last_plan")):
        start_day = thread_blob.get("start_day") or datetime.utcnow().date().isoformat()
        if not thread_blob.get("start_day"):
            memory_store.update_thread_fields(user_id, thread_id, {"start_day": start_day})
        transfer_payload = {
            "user_id": user_id,
            "thread_id": thread_id,
            "user_profile_fields": {
                "level": level,
                "focus": focus,
                "session_minutes": int(user_prof.get("session_minutes") or 10),
                "accessibility": user_prof.get("accessibility") or "voice-friendly",
            },
            "thread_fields": {
                "selected_topic": selected_topic,
                "scenario": thread_blob.get("scenario"),
                "start_day": start_day,
                "current_day": thread_blob.get("current_day", 1),
            },
        }
        if state.get("last_feedback"):
            transfer_payload["last_feedback"] = state.get("last_feedback")
        if state.get("last_rubic_score"):
            transfer_payload["last_rubic_score"] = state.get("last_rubic_score")
        print("[TO PLANNER]", transfer_payload, flush=True)
        planner_agent.invoke({"messages": [HumanMessage(content=json.dumps(transfer_payload, ensure_ascii=False))]})
        thread_blob = memory_store.load_thread(user_id, thread_id) or {}
        state["plan_ready"] = _has_valid_plan(_plan_for_current_day(thread_blob) or thread_blob.get("last_plan"))
        if state.get("plan_ready"):
            state["use_speech"] = True
            return state
    # ===== 2. Gọi supervisor =====
    initial_len = len(state["messages"])
    sup_out = supervisor.invoke({
        "messages": state["messages"],
        "user_id": user_id,
        "thread_id": thread_id,
        "info": state.get("info") or {
            "user_id": user_id,
            "thread_id": thread_id,
        },
    })

    all_msgs = _normalize_messages_from_sup_out(sup_out)
    new_msgs = all_msgs[initial_len:]

    transfer_payload = None

    for m in new_msgs:
        if isinstance(m, AIMessage):
            for c in (getattr(m, "tool_calls", []) or []):
                if c.get("name") == "transfer_to_planner_agent":
                    transfer_payload = c.get("args") or {}
                    break
            if transfer_payload:
                    transfer_payload["user_id"] = user_id
                    transfer_payload["thread_id"] = thread_id
                    # Merge thread_fields with DB values (keep user-chosen topic)
                    thread_blob = memory_store.load_thread(user_id, thread_id) or {}
                    tf = transfer_payload.get("thread_fields") or {}
                    selected_topic = tf.get("selected_topic") or ""
                    if not selected_topic:
                        selected_topic = _infer_topic_from_text(_last_human_text(state.get("messages") or []))
                    transfer_payload["thread_fields"] = {
                        "selected_topic": selected_topic or thread_blob.get("selected_topic"),
                        "scenario": tf.get("scenario") or thread_blob.get("scenario"),
                        "start_day": tf.get("start_day") or thread_blob.get("start_day"),
                        "current_day": thread_blob.get("current_day", 1),
                    }
                    if selected_topic and not thread_blob.get("selected_topic"):
                        memory_store.update_thread_fields(
                            user_id,
                            thread_id,
                            {"selected_topic": selected_topic},
                        )
                    if not transfer_payload["thread_fields"].get("start_day"):
                        transfer_payload["thread_fields"]["start_day"] = datetime.utcnow().date().isoformat()
                        if not thread_blob.get("start_day"):
                            memory_store.update_thread_fields(
                                user_id,
                                thread_id,
                                {"start_day": transfer_payload["thread_fields"]["start_day"]},
                            )
                    if state.get("next_day"):
                        transfer_payload["thread_fields"]["current_day"] = state.get("next_day")
                    prof = transfer_payload.get("user_profile_fields") or {}
                    if prof.get("level"):
                        memory_store.upsert_user_profile(
                            user_id,
                            level=prof.get("level"),
                        )
                    if state.get("last_feedback"):
                        transfer_payload["last_feedback"] = state.get("last_feedback")
                    if state.get("last_rubic_score"):
                        transfer_payload["last_rubic_score"] = state.get("last_rubic_score")
                    print("[TO PLANNER]", transfer_payload, flush=True)
                    planner_agent.invoke({"messages": [
            HumanMessage(
                content=json.dumps(transfer_payload, ensure_ascii=False)
            )
        ]
      })
                    # Mark plan ready for next turn
                    thread_blob = memory_store.load_thread(user_id, thread_id) or {}
                    state["plan_ready"] = _has_valid_plan(_plan_for_current_day(thread_blob) or thread_blob.get("last_plan"))
                    if state["plan_ready"]:
                        state["use_speech"] = True
                        return state
                    if state["plan_ready"]:
                        state["use_speech"] = True
                    break
                  
    # ===== 5. Build new_state =====
    new_state = dict(state)
    new_state["user_id"] = user_id
    new_state["thread_id"] = thread_id
    new_state["messages"] = all_msgs

    last_ai = _last_ai_with_content(all_msgs)
    if last_ai:
        new_state["chat_history"] = state["chat_history"] + [last_ai]

    return new_state
def store_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    """Save conversation and state to database"""
    # CRITICAL: Use original IDs, never generate new ones
    user_id = state.get("user_id")
    thread_id = state.get("thread_id")
    
    if not user_id or not thread_id:
        raise ValueError("user_id and thread_id must be preserved in state")

    # Save learning profile if overrides exist
    incoming_level = state.get("level")
    incoming_focus = state.get("focus")
    incoming_minutes = state.get("session_minutes")
    incoming_access = state.get("accessibility")
    
    if any([incoming_level, incoming_focus, incoming_minutes, incoming_access]):
        memory_store.upsert_user_profile(
            user_id,
            level=incoming_level,
            focus=incoming_focus,
            session_minutes=incoming_minutes,
            accessibility=incoming_access,
        )

    # Save conversation summary
    conv_sum = _simple_concat_summary(state.get("chat_history", []))
    memory_store.update_conv_summary(user_id, thread_id, conv_sum)

    # Save topic and plan
    fields = {}
    if state.get("selected_topic"):
        fields["selected_topic"] = state["selected_topic"]
    if state.get("lesson_plan_json"):
        fields["last_plan"] = state["lesson_plan_json"]

    if fields:
        # keep speech_progress safe (in case some other update path is not $set)
        thread_blob = memory_store.load_thread(user_id, thread_id) or {}
        if "speech_progress" in thread_blob and "speech_progress" not in fields:
            fields["speech_progress"] = thread_blob["speech_progress"]

        memory_store.update_thread_fields(user_id, thread_id, fields)

    # Fallback save if Planner skipped db update
    lesson_plan = state.get("lesson_plan_json") or {}
    meta = lesson_plan.get("meta") or {}
    
    if lesson_plan and meta:
        thread_blob = memory_store.load_thread(user_id, thread_id) or {}
        save_fields = {"last_plan": lesson_plan}
        
        day_index = meta.get("day_index")
        if day_index is not None:
            day_index = int(day_index)
            save_fields[f"last_plan_day_{day_index}"] = lesson_plan
            save_fields["current_day"] = day_index
            
        if _is_empty(thread_blob.get("selected_topic")) and meta.get("selected_topic"):
            save_fields["selected_topic"] = meta["selected_topic"]
            
        if _is_empty(thread_blob.get("scenario")) and meta.get("scenario"):
            save_fields["scenario"] = meta["scenario"]
            
        memory_store.update_thread_fields(user_id, thread_id, save_fields)

    # Reload to sync state
    thread_blob = memory_store.load_thread(user_id, thread_id) or {}
    new_state = dict(state)
    new_state["retrieved_memory"] = {
        "conversation_summary": thread_blob.get("conversation_summary", ""),
        "last_topic": thread_blob.get("last_topic"),
        "last_plan": thread_blob.get("last_plan"),
    }
    new_state["thread_title"] = thread_blob.get("thread_title")
    
    # CRITICAL: Always preserve the original IDs
    new_state["user_id"] = user_id
    new_state["thread_id"] = thread_id

    return new_state

def speech_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Optional speech flow node."""
    if not state.get("use_speech"):
        # Only enter speech flow after user confirms start and a plan exists
        latest_user = _last_human_text(state.get("messages") or [])
        if _is_start_cmd(latest_user):
            thread_blob = memory_store.load_thread(state.get("user_id"), state.get("thread_id")) or {}
            current_day = thread_blob.get("current_day", 1)
            plan_for_day = thread_blob.get(f"last_plan_day_{current_day}")
            if _has_valid_plan(plan_for_day or thread_blob.get("last_plan")):
                state = dict(state)
                state["use_speech"] = True
            else:
                return state
        else:
            return state

    user_id = state.get("user_id")
    thread_id = state.get("thread_id")
    if not user_id or not thread_id:
        return state

    msgs = list(state.get("messages") or [])
    user_text = _last_human_text(msgs)
    out_text = speech_step(user_id, thread_id, user_text=user_text)

    new_state = dict(state)
    ai = AIMessage(content=out_text)
    new_state["messages"] = msgs + [ai]
    new_state["chat_history"] = list(state.get("chat_history") or []) + [ai]
    return new_state

# ===== Build Graph =====
builder = StateGraph(dict)

builder.add_node("handle_request", handle_request)
builder.add_node("retrieve_memories", retrieve_memories)
builder.add_node("executor", executor)
builder.add_node("speech_node", speech_node)
builder.add_node("evaluation_node", evaluation_node)
builder.add_node("store_memory", store_memory)

builder.add_edge(START, "handle_request")
builder.add_edge("handle_request", "retrieve_memories")
builder.add_edge("retrieve_memories", "executor")
builder.add_edge("executor", "speech_node")
builder.add_edge("speech_node", "evaluation_node")
builder.add_edge("evaluation_node", "store_memory")
builder.add_edge("store_memory", END)

graph = builder.compile()
import copy
def run_graph_with_retry(st: dict, attempts: int = 4) -> dict:
    delay = 2.0
    locked_user_id = st.get("user_id")
    locked_thread_id = st.get("thread_id")

    for k in range(attempts):
        # snapshot cho attempt này
        st_try = copy.deepcopy(st)
        # lock lại id trước khi invoke
        st_try["user_id"] = locked_user_id
        st_try["thread_id"] = locked_thread_id

        try:
            out = graph.invoke(st_try)
            # commit: đảm bảo out vẫn giữ id
            out["user_id"] = locked_user_id
            out["thread_id"] = locked_thread_id
            return out
        except Exception as e:
            s = str(e)
            if ("429" in s or "Resource exhausted" in s) and k < attempts - 1:
                time.sleep(delay + random.random())
                delay *= 2
                continue
            raise

# ===== Main CLI =====
if __name__ == "__main__":
    print("=" * 60)
    print("English Coach Supervisor - Interactive Mode")
    print("=" * 60)
    
    # Get user input for IDs - these will be used throughout the session
    USER_ID = input("Enter User ID (default: user123): ").strip() or "user123"
    THREAD_ID = input("Enter Thread ID (default: thread_001): ").strip() or "thread_001"
    
    # Initialize state with FIXED user_id and thread_id
    state: Dict[str, Any] = {
        "user_id": USER_ID,  # This will NEVER change during session
        "thread_id": THREAD_ID,  # This will NEVER change during session
        "chat_history": [],
        "messages": [],
    }

    print(f"\n✓ Session started")
    print(f"  📌 User ID: {USER_ID}")
    print(f"  📌 Thread ID: {THREAD_ID}")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    MAX_MSG = 0
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in {"exit", "quit"}:
                print("\n👋 Goodbye!")
                break

            # Add user message
            user_msg = HumanMessage(content=user_input)
            state["chat_history"].append(user_msg)
            state["messages"].append(user_msg)
            
            # CRITICAL: Ensure IDs are always in state before processing
            state["user_id"] = USER_ID
            state["thread_id"] = THREAD_ID

            # Process with graph
            state = run_graph_with_retry(state)
            
            # CRITICAL: Restore IDs after processing (in case they got lost)
            state["user_id"] = USER_ID
            state["thread_id"] = THREAD_ID
            
            # Ensure keys exist
            state.setdefault("chat_history", [])
            state.setdefault("messages", [])
            
            # Get last AI response
            hist = state.get("chat_history", [])
            last_ai = next((m for m in reversed(hist) if isinstance(m, AIMessage)), None)
            
            if last_ai:
                print(f"\n🤖 Assistant: {str(extract_clean_text(last_ai.content))}\n")
            else:
                print("\n⚠️ (No AI response generated)\n")

            if state.get("should_exit"):
                print("\n👋 Session finished.\n")
                break

            # No trimming: keep full history
                
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            import traceback
            traceback.print_exc()
