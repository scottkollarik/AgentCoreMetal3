from typing import Any, Dict, List, Optional, Union
from .base_agent import BaseAgent
from llm.interface import LLMProvider, OllamaLLMProvider
from utils.logging_interface import AgentLogger, NullLogger
from utils.logging_panel import BaseLoggingPanel
from utils.visualization import VisualizationTools
from utils.search_tools import SearchTools
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.output_parser import OutputParserException
from pydantic import BaseModel, Field
import re
import json
from IPython.display import display, HTML, Markdown
import ipywidgets as widgets
from io import StringIO
import sys
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManagerForLLMRun
import uuid

def get_streaming_callback_manager() -> CallbackManagerForLLMRun:
    """Get a callback manager for streaming output"""
    return CallbackManagerForLLMRun(
        run_id=str(uuid.uuid4()),
        handlers=[StreamingStdOutCallbackHandler()],
        inheritable_handlers=[],
        parent_run_id=None,
        tags=[],
        metadata={}
    )

class ExecutionResult(BaseModel):
    """Result of executing a step"""
    success: bool = Field(description="Whether the step was executed successfully")
    output: str = Field(description="Output or result of the execution")
    error: str = Field(description="Error message if execution failed")
    visualization_type: Optional[str] = Field(description="Type of visualization to use (if any): 'image', 'images'")
    visualization_data: Optional[Dict[str, Any]] = Field(default=None, description="Data for visualization if needed. For images, use format: {'urls': ['url1', 'url2']}")

class ExecutionAgent(BaseAgent):
    """Agent responsible for executing individual steps"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[AgentLogger] = None, llm_provider: Optional[LLMProvider] = None):
        super().__init__(config)
        
        # Initialize logger
        self.debug = config.get("debug", False)
        self.logger = logger if logger else (BaseLoggingPanel(title="Execution Agent Debug Logs") if self.debug else NullLogger())
        
        # Initialize LLM provider
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            if config.get("model_type") == "ollama":
                self.llm_provider = OllamaLLMProvider(
                    model=config.get("model_name", "llama2"),
                    temperature=config.get("temperature", 0.7),
                    max_tokens=config.get("max_tokens", 1000),
                    context_window=config.get("context_window", 8192),
                    logger=self.logger
                )
            else:
                raise ValueError(f"Unsupported model type: {config.get('model_type')}")
        
        # Create output parser
        self.parser = PydanticOutputParser(pydantic_object=ExecutionResult)
        
        # Get format instructions
        format_instructions = self.parser.get_format_instructions()
        
        # Create a more explicit prompt template
        self.execution_prompt = """You are an execution agent that carries out specific steps.

CRITICAL INSTRUCTIONS:
1. Output ONLY a single valid JSON object. No other text, explanations, or commentary.
2. Do not include any capability disclaimers or suggestions.
3. Do not explain what you can or cannot do.
4. Do not add any text before or after the JSON.
5. Do not add comments or additional JSON objects.
6. Keep responses concise and focused.
7. For large data structures, use a simplified format.
8. Use the search tools to find real information:
   - For images: Use SearchTools.search_images()
   - For recipes: Use SearchTools.search_recipes()
   - For trends: Use SearchTools.search_trends()

The JSON must have this exact structure:
{{
  "success": true,
  "output": "The result or output of the execution",
  "error": "",
  "visualization_type": "optional_type",  // One of: "image", "images"
  "visualization_data": {{  // Only include if visualization_type is specified
    "urls": ["url1", "url2"]  // For images
  }}
}}

Requirements:
1. Use double quotes for all keys and string values
2. Include no text before or after the JSON
3. Ensure the JSON is properly formatted with no trailing commas
4. Set success to true only if the step was completed successfully
5. Provide clear output or error messages
6. No comments or additional JSON objects
7. No newlines or extra whitespace in the JSON
8. Keep the output field concise and focused
9. If the step involves images, specify the visualization_type and provide the necessary data
10. Always use search tools to find real information, never make up URLs or data

{format_instructions}

Original Task: {task}
Current Step: {step}
Context: {context}"""
        
        # Create the chain
        self.chain = None  # We'll use generate_with_parser directly
    
    def _extract_step_description(self, step: Union[Dict[str, Any], str]) -> str:
        """Extract step description from various input formats"""
        if isinstance(step, str):
            return step
        
        if isinstance(step, dict):
            # Try common key names for step descriptions
            for key in ['description', 'step', 'task', 'action', 'instruction']:
                if key in step and isinstance(step[key], str):
                    return step[key]
            
            # If no common keys found, try to use the first string value
            for value in step.values():
                if isinstance(value, str):
                    return value
        
        # If we can't extract a description, convert the input to string
        return str(step)
    
    def _handle_visualization(self, result: ExecutionResult) -> None:
        """Handle visualization if specified in the result"""
        if not result.visualization_type:
            return
            
        try:
            if result.visualization_type == "image":
                VisualizationTools.display_image(result.visualization_data["url"])
            elif result.visualization_type == "images":
                VisualizationTools.display_images(result.visualization_data["urls"])
            elif result.visualization_type == "line_chart":
                VisualizationTools.create_line_chart(
                    result.visualization_data["data"],
                    x_field=result.visualization_data.get("x_field", "year"),
                    y_field=result.visualization_data.get("y_field", "popularity"),
                    color_field=result.visualization_data.get("color_field"),
                    title=result.visualization_data.get("title", "Trend Over Time")
                )
            elif result.visualization_type == "bar_chart":
                VisualizationTools.create_bar_chart(
                    result.visualization_data["data"],
                    x_field=result.visualization_data["x_field"],
                    y_field=result.visualization_data["y_field"],
                    title=result.visualization_data.get("title", "Comparison Chart")
                )
            elif result.visualization_type == "trend_chart":
                VisualizationTools.create_trend_chart(
                    result.visualization_data["data"],
                    title=result.visualization_data.get("title", "Trend Analysis")
                )
        except Exception as e:
            if self.debug:
                self.logger.log_error("VISUALIZATION_ERROR", f"Failed to create visualization: {str(e)}")
    
    def _search_for_step(self, step_description: str) -> Dict[str, Any]:
        """Search for information relevant to the step"""
        try:
            # Always try to find relevant images for the step
            images = SearchTools.search_images(step_description)
            if images:
                return {
                    "success": True,
                    "output": f"Found {len(images)} relevant images",
                    "error": "",
                    "visualization_type": "images",
                    "visualization_data": {
                        "urls": [img["url"] for img in images]
                    }
                }
            else:
                return {
                    "success": True,
                    "output": "No relevant images available",
                    "error": "",
                    "visualization_type": None,
                    "visualization_data": None
                }
            
        except Exception as e:
            if self.debug:
                self.logger.log_error("SEARCH_ERROR", f"Failed to search: {str(e)}")
            return {
                "success": False,
                "output": "",
                "error": f"Search failed: {str(e)}",
                "visualization_type": None,
                "visualization_data": None
            }
    
    async def execute(self, step: Union[Dict[str, Any], str], task: Optional[str] = None) -> Dict[str, Any]:
        """Execute a single step"""
        try:
            # Extract step description
            step_description = self._extract_step_description(step)
            
            # Log the step
            if self.debug:
                self.logger.log_event("EXECUTING_STEP", f"Executing step: {step_description}")
                if task:
                    self.logger.log_event("TASK_CONTEXT", f"Original task: {task}")
            
            # First try to search for relevant information
            search_result = self._search_for_step(step_description)
            if search_result["success"] and search_result["visualization_type"]:
                # If we found relevant information, use it
                result = ExecutionResult(**search_result)
                self._handle_visualization(result)
                return search_result
            
            # If no search results or search failed, use LLM
            format_instructions = self.parser.get_format_instructions()
            
            # Generate response with parser
            if self.debug:
                self.logger.log_event("GENERATING_RESPONSE", "Calling LLM provider with parser...")
            
            result = await self.llm_provider.generate_with_parser(
                prompt=self.execution_prompt.format(
                    task=task or "Not provided",
                    step=step_description,
                    context="Previous steps completed successfully",
                    format_instructions=format_instructions
                ),
                parser=self.parser,
                callbacks=get_streaming_callback_manager()
            )
            
            # Handle visualization if specified
            if result.visualization_type:
                self._handle_visualization(result)
            
            # Log the result
            if self.debug:
                self.logger.log_event("EXECUTION_RESULT", f"Result: {result}")
            
            # Convert to dictionary
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "visualization_type": result.visualization_type,
                "visualization_data": result.visualization_data
            }
            
        except Exception as e:
            # Handle any other errors
            if self.debug:
                self.logger.log_error("EXECUTION_ERROR", str(e), e)
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "visualization_type": None,
                "visualization_data": None
            }
    
    async def plan(self, task: str) -> List[Dict[str, Any]]:
        """Execution agent doesn't create plans"""
        raise NotImplementedError("Execution agent doesn't create plans")
    
    async def monitor(self, result: Dict[str, Any]) -> bool:
        """Execution agent doesn't monitor execution"""
        raise NotImplementedError("Execution agent doesn't monitor execution") 