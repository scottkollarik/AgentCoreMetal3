from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class AgentState(BaseModel):
    """Base state model for agents"""
    memory: Dict[str, Any] = {}
    context: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = AgentState()
    
    @abstractmethod
    async def plan(self, task: str) -> List[Dict[str, Any]]:
        """Plan the steps to complete a task"""
        pass
    
    @abstractmethod
    async def execute(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step"""
        pass
    
    @abstractmethod
    async def monitor(self, result: Dict[str, Any]) -> bool:
        """Monitor the execution results"""
        pass
    
    def update_state(self, key: str, value: Any):
        """Update agent state"""
        self.state.memory[key] = value 