from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

class BaseLLM(LLM, ABC):
    """Base class for all LLM implementations"""
    
    @abstractmethod
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LLM API"""
        pass
    
    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Get the type of LLM"""
        pass
    
    @property
    @abstractmethod
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters"""
        pass 