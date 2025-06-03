from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from .base_agent import BaseAgent
import re
from IPython.display import display, HTML, Markdown
import ipywidgets as widgets
from io import StringIO
import sys

class LoggingPanel:
    def __init__(self):
        self.output = widgets.Output()
        self.accordion = widgets.Accordion(children=[self.output])
        self.accordion.set_title(0, 'Debug Logs')
        self.accordion.selected_index = None  # Start collapsed
        self.string_buffer = StringIO()
        self.original_stdout = sys.stdout
        
    def start(self):
        """Start capturing output"""
        sys.stdout = self.string_buffer
        display(self.accordion)
        
    def stop(self):
        """Stop capturing output and display it"""
        sys.stdout = self.original_stdout
        with self.output:
            print(self.string_buffer.getvalue())
        self.string_buffer = StringIO()

class PlanningAgent(BaseAgent):
    """Agent responsible for breaking down tasks into steps"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = ChatOpenAI(
            model_name=config.get("model_name", "gpt-4"),
            temperature=config.get("temperature", 0.7)
        )
        self.debug = config.get("debug", False)
        self.logging_panel = LoggingPanel() if self.debug else None
        
        self.planning_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a planning agent that breaks down complex tasks into manageable steps."),
            ("user", "Task: {task}\nPlease break this down into clear, sequential steps.")
        ])
    
    def _debug_print(self, *args, **kwargs):
        """Print debug information if debug mode is enabled"""
        if self.debug:
            print(*args, **kwargs)
    
    async def plan(self, task: str) -> List[Dict[str, Any]]:
        """Generate a plan for the given task"""
        if self.debug:
            self.logging_panel.start()
            
        try:
            response = await self.llm.ainvoke(
                self.planning_prompt.format_messages(task=task)
            )
            
            # Parse the response into steps
            steps = self._parse_steps(response.content)
            self.state.history.append({"task": task, "plan": steps})
            return steps
        finally:
            if self.debug:
                self.logging_panel.stop()
    
    async def execute(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Planning agent doesn't execute steps"""
        raise NotImplementedError("Planning agent doesn't execute steps")
    
    async def monitor(self, result: Dict[str, Any]) -> bool:
        """Planning agent doesn't monitor execution"""
        raise NotImplementedError("Planning agent doesn't monitor execution")
    
    def _parse_steps(self, content: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured steps"""
        if self.debug:
            print("\n=== Starting Step Parsing ===")
            print(f"Raw content length: {len(content)}")
            print("First 100 chars:", repr(content[:100]))
        
        steps = []
        
        # Split content into lines and clean them
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if self.debug:
            print(f"\nFound {len(lines)} non-empty lines")
            for i, line in enumerate(lines):
                print(f"Line {i}: {repr(line)}")
        
        current_step = None
        current_description = []
        
        for line in lines:
            # Check if this is a step header (e.g., "Step 1: Description")
            if re.match(r'^Step\s+\d+[:.]', line, re.IGNORECASE):
                if self.debug:
                    print(f"\nFound step header: {line}")
                
                # Save previous step if exists
                if current_step is not None:
                    step_desc = " ".join(current_description).strip()
                    if self.debug:
                        print(f"Adding step: {step_desc}")
                    steps.append({
                        "description": step_desc,
                        "status": "pending"
                    })
                
                # Start new step
                current_step = line
                current_description = []
                if self.debug:
                    print(f"Started new step: {current_step}")
            elif current_step is not None:
                # Add to current step's description
                if self.debug:
                    print(f"Adding to description: {line}")
                current_description.append(line)
        
        # Add the last step if exists
        if current_step is not None:
            step_desc = " ".join(current_description).strip()
            if self.debug:
                print(f"\nAdding final step: {step_desc}")
            steps.append({
                "description": step_desc,
                "status": "pending"
            })
        
        if self.debug:
            print(f"\n=== Finished Parsing ===")
            print(f"Total steps found: {len(steps)}")
            for i, step in enumerate(steps, 1):
                print(f"Step {i}: {step['description']}")
        
        return steps
