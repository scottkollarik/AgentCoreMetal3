from typing import Any, Dict, List, Optional
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks import CallbackManager

def get_streaming_callback_manager() -> CallbackManager:
    """Get a callback manager with streaming support"""
    return CallbackManager([StreamingStdOutCallbackHandler()]) 