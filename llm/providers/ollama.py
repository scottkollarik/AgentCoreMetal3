from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from llm.base import BaseLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import ollama
import os
import json
import re
import logging

# Set up logging
logger = logging.getLogger(__name__)

class OllamaLLM(BaseLLM, BaseModel):
    """LangChain wrapper for Ollama LLM"""
    
    model: str = Field(default="llama2", description="The name of the Ollama model to use")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=500, description="Maximum number of tokens to generate")
    context_window: int = Field(default=8192, description="Maximum context window size")
    base_url: str = Field(default="http://localhost:11434", description="Base URL for Ollama API")
    
    def __init__(
        self,
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: int = 500,
        context_window: int = 8192,
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
            base_url=base_url.rstrip('/'),
            **kwargs
        )
        # Set the Ollama host via environment variable
        os.environ["OLLAMA_HOST"] = self.base_url
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Ollama API"""
        try:
            # Add minimal stop sequences for JSON output
            default_stops = [
                "```",  # Stop before code blocks
                "---",  # Stop before markdown separators
                "==="   # Stop before markdown headers
            ]
            
            # Combine default stops with any provided stops
            all_stops = list(set(default_stops + (stop or [])))
            
            # Prepare generation options
            options = {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.context_window,  # Set context window size
                "stop": all_stops,
                **kwargs
            }
            
            # Log the request details
            logger.info("Ollama Request Details:")
            logger.info(f"Model: {self.model}")
            logger.info(f"Base URL: {self.base_url}")
            logger.info(f"Options: {json.dumps(options, indent=2)}")
            logger.info("Prompt:")
            logger.info("-" * 80)
            logger.info(prompt)
            logger.info("-" * 80)
            
            # Generate response with streaming
            full_response = ""
            for chunk in ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
                options=options
            ):
                if "error" in chunk:
                    logger.error(f"Ollama API error: {chunk['error']}")
                    raise ValueError(f"Ollama API error: {chunk['error']}")
                if "response" in chunk:
                    chunk_text = chunk["response"]
                    full_response += chunk_text
                    if run_manager:
                        run_manager.on_llm_new_token(chunk_text)
            
            if not full_response:
                logger.error("Empty response from Ollama API")
                raise ValueError("Empty response from Ollama API")
            
            # Clean up the response
            full_response = full_response.strip()
            
            # Try to extract the first valid JSON object
            json_match = re.search(r'\{[\s\S]*?\}(?=\s*\{|$)', full_response)
            if json_match:
                full_response = json_match.group(0)
            
            # Log the response
            logger.info("Ollama Response:")
            logger.info("-" * 80)
            logger.info(full_response)
            logger.info("-" * 80)
            
            # Return the response
            return full_response
            
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise ValueError(f"Error calling Ollama API: {str(e)}")
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
            "base_url": self.base_url
        } 