from typing import Optional, Dict, Any
from datetime import datetime
from pymongo.collection import Collection
from pymongo import ReturnDocument

class MemoryStoreMongo:
    def __init__(self, user_profiles_col: Collection, threads_col: Collection):
        self.user_profiles = user_profiles_col
        self.threads = threads_col

        self.user_profiles.create_index([("user_id", 1)], unique=True, background=True)
        self.threads.create_index([("user_id", 1), ("thread_id", 1)], unique=True, background=True)

    # -------- LOAD --------
    def load_user_profile(self, user_id: str) -> Dict[str, Any]:
        doc = self.user_profiles.find_one({"user_id": user_id})
        if not doc:
            # user lần đầu -> trả về profile rỗng hợp lệ
            return {
                "level": None,            # "mới bắt đầu" / "trung bình" / "khá" hoặc A1/A2...
                "focus": None,            # "listening" | "speaking" | "both"
                "session_minutes": None,  # int, vd 10
                "accessibility": None,    # "voice-friendly" ...
                "session_records": [],
            }
        return {
            "level": doc.get("level"),
            "focus": doc.get("focus"),
            "session_minutes": doc.get("session_minutes"),
            "accessibility": doc.get("accessibility"),
            "session_records": doc.get("session_records", []),
        }

    def load_thread(self, user_id: str, thread_id: str) -> Dict[str, Any]:
        doc = self.threads.find_one({"user_id": user_id, "thread_id": thread_id})
        if not doc:
            # thread lần đầu -> trả về thread rỗng hợp lệ
            return {
                "thread_title": None,
                "conversation_summary": "",
                "selected_topic": None,
                "scenario": None,
                "start_day": None,
                "current_day": 1,
                "last_day_result": None,
                "last_plan": None,
                "speech_progress": None,
                "last_topic": None,
                "last_usage_date": None,
                "usage_day_count": 0,
                "session_attempts_by_day": {},
                "last_session_logged": None,
            }
        return {
            "thread_title": doc.get("thread_title"),
            "conversation_summary": doc.get("conversation_summary", ""),
            "selected_topic": doc.get("selected_topic"),
            "scenario": doc.get("scenario"),
            "start_day": doc.get("start_day"),
            "current_day": doc.get("current_day", 1),
            "last_day_result": doc.get("last_day_result"),
            "last_plan": doc.get("last_plan"),
            "speech_progress": doc.get("speech_progress"),
            "last_topic": doc.get("last_topic"),
            "last_usage_date": doc.get("last_usage_date"),
            "usage_day_count": doc.get("usage_day_count", 0),
            "session_attempts_by_day": doc.get("session_attempts_by_day", {}),
            "last_session_logged": doc.get("last_session_logged"),
        }
    # -------- UPSERT / UPDATE --------
    def upsert_user_profile(
        self,
        user_id: str,
        level: Optional[str] = None,
        focus: Optional[str] = None,
        session_minutes: Optional[int] = None,
        accessibility: Optional[str] = None,
    ) -> None:
        update_fields: Dict[str, Any] = {}
        if level is not None:
            update_fields["level"] = level
        if focus is not None:
            update_fields["focus"] = focus
        if session_minutes is not None:
            update_fields["session_minutes"] = session_minutes
        if accessibility is not None:
            update_fields["accessibility"] = accessibility

        if not update_fields:
            return

        update_fields["updated_at"] = datetime.utcnow()

        self.user_profiles.find_one_and_update(
            {"user_id": user_id},
            {"$set": update_fields},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def update_conv_summary(self, user_id: str, thread_id: str, new_summary: str) -> None:
        self.threads.find_one_and_update(
            {"user_id": user_id, "thread_id": thread_id},
            {"$set": {"conversation_summary": new_summary, "updated_at": datetime.utcnow()}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def set_thread_title(self, user_id: str, thread_id: str, title: str) -> None:
        self.threads.find_one_and_update(
            {"user_id": user_id, "thread_id": thread_id},
            {"$set": {"thread_title": title, "updated_at": datetime.utcnow()}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def get_thread_title(self, user_id: str, thread_id: str) -> Optional[str]:
        doc = self.threads.find_one(
            {"user_id": user_id, "thread_id": thread_id},
            {"thread_title": 1},
        )
        return (doc or {}).get("thread_title")

    # -------- NEW: thread fields for coaching --------
    def set_last_topic(self, user_id: str, thread_id: str, last_topic: str) -> None:
        self.threads.find_one_and_update(
            {"user_id": user_id, "thread_id": thread_id},
            {"$set": {"last_topic": last_topic, "updated_at": datetime.utcnow()}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def set_last_plan(self, user_id: str, thread_id: str, last_plan: Any) -> None:
        """
        last_plan: có thể là dict JSON hoặc string JSON.
        Khuyến nghị: lưu dict để retriever/analytics sau này dễ dùng.
        """
        self.threads.find_one_and_update(
            {"user_id": user_id, "thread_id": thread_id},
            {"$set": {"last_plan": last_plan, "updated_at": datetime.utcnow()}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def update_thread_fields(self, user_id: str, thread_id: str, fields: Dict[str, Any]) -> None:
        """
        Helper chung để bạn set nhiều field một lần (giống nhu cầu bạn nói).
        """
        if not fields:
            return
        fields = dict(fields)
        fields["updated_at"] = datetime.utcnow()
        self.threads.find_one_and_update(
            {"user_id": user_id, "thread_id": thread_id},
            {"$set": fields},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def append_session_record(self, user_id: str, record: Dict[str, Any]) -> None:
        if not record:
            return
        self.user_profiles.find_one_and_update(
            {"user_id": user_id},
            {
                "$push": {"session_records": record},
                "$setOnInsert": {"created_at": datetime.utcnow()},
                "$set": {"updated_at": datetime.utcnow()},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
