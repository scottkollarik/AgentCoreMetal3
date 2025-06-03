from typing import Any, Dict, List
from .base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from memory.error_memory import ErrorMemory

class ExecutionAgent(BaseAgent):
    """Agent responsible for executing planned steps"""
    
    def __init__(self, config: Dict[str, Any], tools: List[Tool] = None, error_memory: ErrorMemory = None):
        super().__init__(config)
        self.llm = ChatOpenAI(
            model_name=config.get("model_name", "gpt-4"),
            temperature=config.get("temperature", 0.5)
        )
        
        self.tools = tools or []
        self.error_memory = error_memory
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self._create_prompt()
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )
    
    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an execution agent that carries out tasks step by step.
            You have access to the following tools: {tools}
            
            Use the following format:
            
            Step: the step you need to execute
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final output of executing the step
            
            Begin!"""),
            ("user", "Execute the following step: {step}"),
            ("assistant", "{agent_scratchpad}")
        ])
    
    async def plan(self, task: str) -> List[Dict[str, Any]]:
        """Execution agent doesn't create plans"""
        raise NotImplementedError("Execution agent doesn't create plans")
    
    async def execute(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step"""
        try:
            # Get tool names for the prompt
            tool_names = [tool.name for tool in self.tools]
            
            # Format tools for the prompt
            tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
            
            result = await self.agent_executor.ainvoke({
                "step": step["description"],
                "tool_names": ", ".join(tool_names),
                "tools": tools_str
            })
            
            self.state.history.append({
                "step": step,
                "result": result,
                "status": "completed"
            })
            
            return {
                "status": "success",
                "result": result,
                "step": step
            }
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "step": step
            }
            
            # Store error in error memory if available
            if self.error_memory:
                self.error_memory.add_error(
                    str(e),
                    context={
                        "step": step,
                        "error_type": type(e).__name__,
                        "agent": "execution_agent"
                    }
                )
            
            return error_result
    
    async def monitor(self, result: Dict[str, Any]) -> bool:
        """Basic monitoring of execution results"""
        if result["status"] == "success":
            return True
        return False
    
    def add_tool(self, tool: Tool):
        """Add a new tool to the agent"""
        self.tools.append(tool)
        # Recreate agent with new tools
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self._create_prompt()
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        ) 