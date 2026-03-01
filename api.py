import os
import sys
import threading
import uuid
import hashlib
import hmac
from datetime import datetime
from typing import Dict, Any, Tuple
from src.utils.extract_text import extract_clean_text
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from src.tools.schema import (
    ChatRequest,
    ChatResponse,
    LoginRequest,
    LoginResponse,
    SignupRequest,
    SignupResponse,
    ValidateIntentRequest,
    ValidateIntentResponse,
)
load_dotenv()
from src.agents.supervisor import run_graph_with_retry
from src.agents.validation_agent import validate_start_intent
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from src.agents.supervisor import run_graph_with_retry  # noqa: E402

app = FastAPI(title="chat-lis-speak API", version="0.1.0")

_state_lock = threading.Lock()
_session_states: Dict[Tuple[str, str], Dict[str, Any]] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mongo = MongoClient(os.getenv("MONGO_URI"))
db = mongo[os.getenv("MONGO_DB", "english_coach_db")]
user_accounts = db["user_accounts"]
user_profiles = db["user_profiles"]
threads = db["threads"]
user_accounts.create_index([("account", 1)], unique=True, background=True)
user_accounts.create_index([("user_id", 1)], unique=True, background=True)


def _hash_password(password: str, salt: bytes | None = None, iterations: int = 200_000) -> Dict[str, Any]:
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return {
        "salt": salt.hex(),
        "hash": dk.hex(),
        "iterations": iterations,
    }


def _verify_password(password: str, record: Dict[str, Any]) -> bool:
    salt_hex = record.get("salt") or ""
    hash_hex = record.get("hash") or ""
    iterations = int(record.get("iterations") or 200_000)
    if not salt_hex or not hash_hex:
        return False
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk.hex(), hash_hex)


def _plan_ready(user_id: str, thread_id: str) -> bool:
    doc = threads.find_one({"user_id": user_id, "thread_id": thread_id}) or {}
    if not doc:
        return False
    current_day = int(doc.get("current_day", 1))
    if doc.get(f"last_plan_day_{current_day}"):
        return True
    return bool(doc.get("last_plan"))


def _init_state(user_id: str, thread_id: str) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "chat_history": [],
        "messages": [],
    }


def _get_state(user_id: str, thread_id: str, reset: bool) -> Dict[str, Any]:
    key = (user_id, thread_id)
    with _state_lock:
        if reset or key not in _session_states:
            _session_states[key] = _init_state(user_id, thread_id)
        return _session_states[key]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    state = _get_state(req.user_id, req.thread_id, req.reset)

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is empty")
    req_id = request.headers.get("x-request-id", "")
    print(f"[API] chat req id={req_id} user_id={req.user_id} thread_id={req.thread_id} msg={repr(req.message)}", flush=True)

    user_msg = HumanMessage(content=req.message.strip())
    state["chat_history"].append(user_msg)
    state["messages"].append(user_msg)
    state["user_id"] = req.user_id
    state["thread_id"] = req.thread_id

    try:
        state = run_graph_with_retry(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    state["user_id"] = req.user_id
    state["thread_id"] = req.thread_id
    state.setdefault("chat_history", [])
    state.setdefault("messages", [])

    hist = state.get("chat_history", [])
    last_ai = next((m for m in reversed(hist) if isinstance(m, AIMessage)), None)
    if not last_ai:
        raise HTTPException(status_code=500, detail="no AI response generated")
    print(f"[API] last_ai.content={repr(last_ai.content)}", flush=True)

    return ChatResponse(
        user_id=req.user_id,
        thread_id=req.thread_id,
        assistant_message=str(extract_clean_text(last_ai.content)),
        should_exit=bool(state.get("should_exit")),
    )


@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest) -> LoginResponse:
    account = req.account.strip().lower()
    if not account:
        raise HTTPException(status_code=400, detail="account is empty")

    doc = user_accounts.find_one({"account": account})
    if not doc:
        raise HTTPException(status_code=404, detail="account not found")

    if not doc or "password" not in doc:
        raise HTTPException(status_code=500, detail="account record invalid")

    if not _verify_password(req.password, doc["password"]):
        raise HTTPException(status_code=401, detail="invalid password")

    return LoginResponse(
        user_id=doc.get("user_id", ""),
        account=doc.get("account", account),
        created=False,
    )


@app.post("/signup", response_model=SignupResponse)
def signup(req: SignupRequest) -> SignupResponse:
    account = req.account.strip().lower()
    if not account:
        raise HTTPException(status_code=400, detail="account is empty")

    if user_accounts.find_one({"account": account}):
        raise HTTPException(status_code=409, detail="account already exists")

    user_id = uuid.uuid4().hex
    password_record = _hash_password(req.password)
    now = datetime.utcnow()
    payload = {
        "user_id": user_id,
        "account": account,
        "password": password_record,
        "created_at": now,
        "updated_at": now,
    }
    try:
        user_accounts.insert_one(payload)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="account already exists")

    return SignupResponse(user_id=user_id, account=account)


@app.get("/progress")
def progress(user_id: str) -> Dict[str, int]:
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    doc = user_profiles.find_one({"user_id": user_id}, {"session_records": 1})
    records = (doc or {}).get("session_records") or []
    max_day = 0
    for r in records:
        try:
            day = int(r.get("day_index", 0))
        except Exception:
            day = 0
        if day > max_day:
            max_day = day
    current_day = min(max_day + 1, 4) if max_day else 1
    return {"completed_days": max_day, "current_day": current_day}


@app.get("/daily-status")
def daily_status(user_id: str) -> Dict[str, Any]:
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    doc = user_profiles.find_one({"user_id": user_id}, {"session_records": 1})
    records = (doc or {}).get("session_records") or []
    today = datetime.utcnow().date().isoformat()
    completed_today = False
    has_rubric = False
    for r in records:
        if (r.get("date") or "") == today:
            completed_today = True
            rubric = r.get("rubric_scores") or {}
            if isinstance(rubric, dict) and rubric:
                has_rubric = True
    return {"completed_today": completed_today, "has_rubric": has_rubric, "date": today}
    
@app.post("/validate-intent", response_model=ValidateIntentResponse)
def validate_intent(req: ValidateIntentRequest) -> ValidateIntentResponse:
    result = validate_start_intent(req.message)
    return ValidateIntentResponse(
        user_id=req.user_id,
        should_start=bool(result["should_start"]),
        confidence=float(result["confidence"]),
        reason=str(result["reason"]),
        normalized_message=str(result["normalized_message"]),
    )

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, reload=False)
