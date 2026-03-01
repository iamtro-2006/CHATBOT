# memory/memory_agent.py
from langchain_core.messages import SystemMessage
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from utils.load_model import load_chat_model

from tools.memory_tools import (
    db_get_thread,
    db_upsert_user_profile,
    db_update_thread_fields,
)

memory_model = load_chat_model(
    "google_vertexai/gemini-2.5-flash",
    tags=["memory_agent"],
    temperature=0.0,
)

MEMORY_AGENT_PROMPT = r"""
Bạn là MEMORY_AGENT. Nhiệm vụ DUY NHẤT: lưu dữ liệu vào MongoDB bằng tools.

INPUT bạn nhận sẽ là payload JSON do Supervisor gửi, thường có dạng:
{
  "user_id": "...",
  "thread_id": "...",
  "user_profile_fields": { "level": "...", "focus": "...", "session_minutes": 10, "accessibility": "voice-friendly" },
  "thread_fields": { "selected_topic": "...", "scenario": "...", "start_day": "...", "current_day": 1 }
}

QUY TẮC CỨNG:
1) Không được bịa user_id/thread_id. Luôn dùng đúng user_id + thread_id từ payload.
2) Luôn gọi db_get_thread(user_id, thread_id) trước để biết DB đang có gì.
3) Lưu user_profile_fields bằng db_upsert_user_profile (bắt buộc).
4) Lưu thread_fields bằng db_update_thread_fields, NHƯNG:
   - chỉ set start_day nếu DB đang thiếu start_day
   - chỉ set current_day nếu DB đang thiếu current_day
   - selected_topic/scenario: set nếu payload có và DB đang thiếu
5) Sau khi lưu xong, gọi transfer_back_to_supervisor.
6) Không trả JSON dài. Chỉ 1 câu ngắn xác nhận đã lưu.
TUYỆT ĐỐI KHÔNG tự tạo user_id hoặc thread_id.
"""

memory_agent = create_react_agent(
    model=memory_model,
    tools=[db_get_thread, db_upsert_user_profile, db_update_thread_fields],
    prompt=SystemMessage(content=MEMORY_AGENT_PROMPT),
    name="memory_agent",
)
