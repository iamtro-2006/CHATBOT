import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from src.agents import supervisor, planner  # noqa: E402


def _render_prompt(template: str, **kwargs: str) -> str:
    rendered = template
    for key, value in kwargs.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _call_model(model, text: str) -> str:
    response = model.invoke(text)
    return getattr(response, "content", str(response))


def main() -> None:
    topics = [
        "Daily routines and personal info",
        "Food, drinks, and ordering politely",
        "Getting around town and travel basics",
    ]
    conversation_state = {"topics_list": topics, "selected_topic": None, "day_index": 1}
    user_profile = {
        "estimated_level": "A1",
        "english_only_mode": False,
        "preferences": {},
        "recurring_errors": [],
    }
    memory_snapshot = {"due_vocab": [], "last_session_summary": "", "recent_errors": []}

    print("Supervisor chat started. Type 'exit' to stop.")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        context = "\n".join(
            [
                f"user_utterance_text: {user_input}",
                f"conversation_state JSON: {json.dumps(conversation_state)}",
                f"user_profile JSON: {json.dumps(user_profile)}",
                f"memory_snapshot JSON: {json.dumps(memory_snapshot)}",
            ]
        )
        supervisor_prompt = f"{supervisor.prompt}\n\n{context}"
        raw = _call_model(supervisor.model, supervisor_prompt)

        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            print("Supervisor: Invalid JSON response.")
            print(raw)
            continue

        response_type = decision.get("type")
        if response_type == "tool_call":
            tool_args = decision.get("tool_args", {})
            topic = tool_args.get("topic")
            day_index = tool_args.get("day_index", conversation_state["day_index"])
            profile_with_day = dict(user_profile)
            profile_with_day["day_index"] = day_index

            planner_prompt = _render_prompt(
                planner.prompt,
                TOPIC=topic or "",
                PROFILE_JSON=json.dumps(profile_with_day),
                MEMORY_JSON=json.dumps(memory_snapshot),
            )
            planner_output = _call_model(planner.model, planner_prompt)
            print("Planner output:")
            print(planner_output)
            conversation_state["selected_topic"] = topic
            conversation_state["day_index"] = day_index
        elif response_type == "clarify":
            print(decision.get("question_vi") or "")
            print(decision.get("question_en") or "")
        elif response_type == "direct_response":
            print(decision.get("message_vi") or "")
            print(decision.get("message_en") or "")
        else:
            print("Supervisor: Unknown response type.")
            print(raw)


if __name__ == "__main__":
    main()
