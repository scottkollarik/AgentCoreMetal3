from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent
from llm.interface import LLMProvider, OllamaLLMProvider
from utils.logging_interface import AgentLogger, NullLogger
from utils.logging_panel import BaseLoggingPanel
from utils.callback_handlers import get_streaming_callback_manager
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.output_parser import OutputParserException
from pydantic import BaseModel, Field
import re
from IPython.display import display, HTML, Markdown
import ipywidgets as widgets
from io import StringIO
import sys
import json

class Step(BaseModel):
    """A single step in the plan"""
    step: int = Field(description="The step number")
    description: str = Field(description="Description of what needs to be done")

class Plan(BaseModel):
    """A plan consisting of multiple steps"""
    steps: List[Step] = Field(description="List of steps in the plan")

class PlanningAgent(BaseAgent):
    """Agent responsible for breaking down tasks into steps"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[AgentLogger] = None, llm_provider: Optional[LLMProvider] = None):
        super().__init__(config)
        
        # Initialize logger
        self.debug = config.get("debug", False)
        self.logger = logger if logger else (BaseLoggingPanel(title="Planning Agent Debug Logs") if self.debug else NullLogger())
        
        # Initialize LLM provider
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            if config.get("model_type") == "ollama":
                self.llm_provider = OllamaLLMProvider(
                    model=config.get("model_name", "llama2"),
                    temperature=config.get("temperature", 0.7),
                    max_tokens=config.get("max_tokens", 500),
                    context_window=config.get("context_window", 8192),
                    logger=self.logger
                )
            else:
                raise ValueError(f"Unsupported model type: {config.get('model_type')}")
        
        # Create output parser
        self.parser = PydanticOutputParser(pydantic_object=Plan)
        
        # Create a more explicit prompt template
        self.planning_prompt = """You are a task planning agent. Your job is to break down tasks into clear, sequential steps.

CRITICAL INSTRUCTIONS:
1. Output ONLY a single valid JSON object. No other text, explanations, or commentary.
2. Do not include any capability disclaimers or suggestions.
3. Do not explain what you can or cannot do.
4. Do not add any text before or after the JSON.
5. Do not add comments or additional JSON objects.
6. All steps must be in a single JSON object.

The JSON must have this exact structure:
{{
  "steps": [
    {{
      "step": 1,
      "description": "First step description"
    }},
    {{
      "step": 2,
      "description": "Second step description"
    }}
  ]
}}

Requirements:
1. Use double quotes for all keys and string values
2. Include no text before or after the JSON
3. Ensure the JSON is properly formatted with no trailing commas
4. Each step must have a "step" number and "description"
5. All steps must be in a single JSON object
6. No comments or additional JSON objects

Task: {task}"""
        
        # Create the chain
        self.chain = None  # We'll use generate_with_parser directly
        
    async def plan(self, task: str) -> List[Dict[str, Any]]:
        """Generate a plan for the given task"""
        try:
            # Validate task is not empty
            if not task or not task.strip():
                raise ValueError("Task cannot be empty")
            
            # Get format instructions
            format_instructions = self.parser.get_format_instructions()
            
            # Generate and parse the response
            self.logger.log_event("MODEL_REQUEST", f"Sending request to {self.llm_provider.get_model_name()}")
            
            # Log the full prompt for debugging
            if self.debug:
                full_prompt = self.planning_prompt.format(task=task)
                self.logger.log_event("FULL_PROMPT", full_prompt)
            
            # Generate with streaming
            result = await self.llm_provider.generate_with_parser(
                prompt=self.planning_prompt.format(task=task),
                parser=self.parser,
                callbacks=get_streaming_callback_manager()
            )
            
            # Validate the result
            if not result or not result.steps:
                raise ValueError("Generated plan is empty")
            
            # Validate step numbers are sequential
            step_numbers = [step.step for step in result.steps]
            if step_numbers != list(range(1, len(result.steps) + 1)):
                raise ValueError("Step numbers must be sequential starting from 1")
            
            # Validate step descriptions are not empty
            for step in result.steps:
                if not step.description or not step.description.strip():
                    raise ValueError(f"Step {step.step} description cannot be empty")
            
            # Log the result
            if self.debug:
                self.logger.log_event("PLAN_GENERATED", f"Generated plan with {len(result.steps)} steps")
                for step in result.steps:
                    self.logger.log_event("STEP", f"Step {step.step}: {step.description}")
            
            # Convert to list of dictionaries for easier access
            return [{"step": step.step, "description": step.description} for step in result.steps]
            
        except OutputParserException as e:
            # Handle parsing errors
            if self.debug:
                self.logger.log_error("PARSER_ERROR", str(e), e)
                if hasattr(e, 'llm_output'):
                    self.logger.log_model_response("Raw Output", e.llm_output)
            raise
            
        except ValueError as e:
            # Handle LLM provider errors
            if self.debug:
                self.logger.log_error("LLM_ERROR", str(e), e)
            raise
            
        except Exception as e:
            # Handle any other errors
            if self.debug:
                self.logger.log_error("UNEXPECTED_ERROR", str(e), e)
            raise
            
    async def execute(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Planning agent doesn't execute steps"""
        raise NotImplementedError("Planning agent doesn't execute steps")
    
    async def monitor(self, result: Dict[str, Any]) -> bool:
        """Planning agent doesn't monitor execution"""
        raise NotImplementedError("Planning agent doesn't monitor execution")
