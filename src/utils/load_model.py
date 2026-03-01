from typing import List, Optional
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
import vertexai
import os
vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEXAI_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("VERTEXAI_LOCATION") or "us-central1",
)

def load_chat_model(
    fully_specified_name: str, tags: Optional[List[str]] = None, temperature: float = 0, disable_streaming=True
) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(
        model, model_provider=provider, tags=tags, temperature=temperature, disable_streaming=disable_streaming
    )