from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain.llms.base import LLM
from langchain.schema.output_parser import OutputParserException
from .providers.ollama import OllamaLLM
from langchain_openai import ChatOpenAI
import logging
import re
from langchain.schema.output_parser import BaseOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from utils.logging_interface import AgentLogger

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        pass
        
    @abstractmethod
    async def generate_with_parser(self, prompt: str, parser: Any, **kwargs) -> Any:
        """Generate text and parse it with the given parser"""
        pass
        
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model being used"""
        pass

class OllamaLLMProvider(LLMProvider):
    """Provider for Ollama LLM"""
    
    def __init__(
        self,
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        context_window: int = 8192,
        logger: Optional[AgentLogger] = None
    ):
        self.llm = OllamaLLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window
        )
        self.logger = logger
        self._model_name = model
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama"""
        try:
            response = await self.llm.agenerate(prompt, **kwargs)
            return response.generations[0][0].text
        except Exception as e:
            raise ValueError(f"Ollama generation failed: {str(e)}")
            
    async def generate_with_parser(
        self,
        prompt: str,
        parser: BaseOutputParser,
        callbacks: Optional[List[BaseCallbackHandler]] = None
    ) -> Any:
        """Generate text with a parser"""
        try:
            # Generate the response
            response = await self.llm.agenerate([prompt], callbacks=callbacks)
            
            # Log the raw response for debugging
            if self.logger is not None:
                self.logger.log_event("RAW_RESPONSE", str(response))
            
            # Get the generated text
            generated_text = response.generations[0][0].text.strip()
            
            # Log the generated text for debugging
            if self.logger is not None:
                self.logger.log_event("GENERATED_TEXT", generated_text)
            
            # Try to extract JSON if it's wrapped in other text
            json_match = re.search(r'\{[\s\S]*\}', generated_text)
            if json_match:
                generated_text = json_match.group(0)
                if self.logger is not None:
                    self.logger.log_event("EXTRACTED_JSON", generated_text)
            
            # Parse the response
            try:
                parsed_result = parser.parse(generated_text)
                return parsed_result
            except Exception as e:
                if self.logger is not None:
                    self.logger.log_error("PARSE_ERROR", f"Failed to parse response: {str(e)}", e)
                    self.logger.log_event("FAILED_TEXT", generated_text)
                raise
                
        except Exception as e:
            if self.logger is not None:
                self.logger.log_error("GENERATION_ERROR", f"Failed to generate response: {str(e)}", e)
            raise
            
    def get_model_name(self) -> str:
        return self._model_name

class OpenAILLMProvider(LLMProvider):
    """Provider for OpenAI LLM models"""
    
    def __init__(self, model: str = "gpt-4", **kwargs):
        self.llm = ChatOpenAI(
            model_name=model,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000)
        )
        self._model_name = model
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI"""
        try:
            response = await self.llm.agenerate(prompt, **kwargs)
            return response.generations[0][0].text
        except Exception as e:
            raise ValueError(f"OpenAI generation failed: {str(e)}")
            
    async def generate_with_parser(self, prompt: str, parser: Any, **kwargs) -> Any:
        """Generate text and parse it with the given parser"""
        try:
            response = await self.llm.agenerate(prompt, **kwargs)
            return parser.parse(response.generations[0][0].text)
        except OutputParserException as e:
            raise ValueError(f"Failed to parse OpenAI response: {str(e)}")
        except Exception as e:
            raise ValueError(f"OpenAI generation failed: {str(e)}")
            
    def get_model_name(self) -> str:
        return self._model_name 