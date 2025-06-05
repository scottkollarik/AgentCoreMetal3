from typing import Dict, Any, Optional, List
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_openai import ChatOpenAI
from .ollama_llm import OllamaLLM
from pydantic import BaseModel, Field

class FallbackLLM(LLM, BaseModel):
    """LLM that falls back to OpenAI if Ollama fails"""
    
    ollama: OllamaLLM = Field(description="Primary Ollama LLM")
    openai: ChatOpenAI = Field(description="Fallback OpenAI LLM")
    
    def __init__(self, ollama_config: Dict[str, Any], openai_config: Dict[str, Any]):
        super().__init__(
            ollama=OllamaLLM(**ollama_config),
            openai=ChatOpenAI(**openai_config)
        )
    
    @property
    def _llm_type(self) -> str:
        return "fallback"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Try Ollama first, fall back to OpenAI if it fails"""
        try:
            return self.ollama._call(prompt, stop, run_manager, **kwargs)
        except Exception as e:
            print(f"Ollama failed, falling back to OpenAI: {str(e)}")
            return self.openai._call(prompt, stop, run_manager, **kwargs)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters"""
        return {
            "ollama": self.ollama._identifying_params,
            "openai": self.openai._identifying_params
        } 