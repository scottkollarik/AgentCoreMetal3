from typing import Dict, Any, Optional, List, TypedDict, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib
from datetime import datetime

class ComponentType(Enum):
    """Types of components in the system"""
    AGENT = "agent"
    TOOL = "tool"
    MEMORY = "memory"
    MODEL = "model"
    ORCHESTRATOR = "orchestrator"

class LocalModelConfig(TypedDict):
    """Configuration for locally hosted models (e.g., Ollama)"""
    model_name: str
    model_type: str
    temperature: float
    max_tokens: int
    base_url: str

class RemoteModelConfig(TypedDict):
    """Configuration for remote model APIs (e.g., OpenAI)"""
    model_name: str
    api_key: str
    api_base: str
    temperature: float
    max_tokens: int

class ComponentData(TypedDict):
    """Additional data for components"""
    input_data: Dict[str, Any]  # Input data for the component
    output_data: Dict[str, Any]  # Output data from the component
    state: Dict[str, Any]  # Current state of the component
    model_config: Union[LocalModelConfig, RemoteModelConfig]  # Model configuration
    metrics: Dict[str, float]  # Performance metrics (e.g., latency, token usage)
    error: Optional[str]  # Error information if any

def generate_short_id() -> str:
    """Generate a short unique identifier"""
    return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]

@dataclass
class ComponentContext:
    """Context for managing component state and relationships"""
    
    # Component identification
    component_type: ComponentType
    name: str
    version: str
    id: str = field(default_factory=generate_short_id)
    
    # Relationships
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    intent_executor_ids: List[str] = field(default_factory=list)
    
    # Data management
    component_data: ComponentData = field(default_factory=lambda: {
        "input_data": {},
        "output_data": {},
        "state": {},
        "model_config": {},
        "metrics": {},
        "error": None
    })
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def full_id(self) -> str:
        """Get the full component identifier"""
        return f"{self.component_type.value}-{self.name}-{self.id}"
    
    def to_json(self) -> Dict[str, Any]:
        """Convert context to JSON-serializable format"""
        return {
            "component_type": self.component_type.value,
            "name": self.name,
            "version": self.version,
            "id": self.id,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "intent_executor_ids": self.intent_executor_ids,
            "component_data": self.component_data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def create_child_context(self, name: str, **kwargs) -> "ComponentContext":
        """Create a child context with inherited properties"""
        child = ComponentContext(
            component_type=self.component_type,
            name=name,
            version=self.version,
            parent_id=self.id,
            **kwargs
        )
        self.child_ids.append(child.id)
        return child
    
    def update_component_data(self, **kwargs) -> None:
        """Update component data with validation"""
        for key, value in kwargs.items():
            if key not in self.component_data:
                raise ValueError(f"Invalid data field: {key}")
            self.component_data[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_component_data(self, key: str) -> Any:
        """Get component data with validation"""
        if key not in self.component_data:
            raise ValueError(f"Invalid data field: {key}")
        return self.component_data[key]
    
    def add_intent_executor(self, executor_id: str) -> None:
        """Add an intent executor with duplicate prevention"""
        if executor_id not in self.intent_executor_ids:
            self.intent_executor_ids.append(executor_id)
            self.updated_at = datetime.utcnow() 