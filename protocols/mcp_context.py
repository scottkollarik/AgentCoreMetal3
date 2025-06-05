from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib
from datetime import datetime

class ComponentType(str, Enum):
    AGENT = "agent"
    TOOL = "tool"
    MEMORY = "memory"
    MODEL = "model"
    ORCHESTRATOR = "orchestrator"

def generate_short_id() -> str:
    """Generate a short unique identifier"""
    return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]

@dataclass
class MCPContext:
    """Base class for standardized MCP context configuration"""
    
    component_type: ComponentType
    name: str
    version: str = "1.0.0"
    model_config: Optional[Dict[str, Any]] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Flow and component identifiers
    flow_id: Optional[str] = None  # ID of the overall flow/chain
    run_id: Optional[str] = None   # ID of this specific run
    component_id: str = field(default_factory=generate_short_id)  # Unique ID for this component instance
    parent_id: Optional[str] = None  # ID of parent component if part of a chain
    
    # Component relationship tracking
    prior_component_ids: List[str] = field(default_factory=list)  # List of components that executed before this one
    intent_executor_ids: List[str] = field(default_factory=list)  # List of potential executors for this component's intent
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def __post_init__(self):
        """Initialize after dataclass creation"""
        if not self.flow_id:
            self.flow_id = generate_short_id()
        if not self.run_id:
            self.run_id = generate_short_id()
            
    @property
    def full_id(self) -> str:
        """Get the full component identifier"""
        parts = [self.flow_id, self.run_id, self.component_id]
        if self.parent_id:
            parts.insert(2, self.parent_id)
        return ":".join(parts)
    
    def to_json(self) -> Dict[str, Any]:
        """Convert context to JSON-serializable dictionary"""
        return {
            "component_type": self.component_type.value,
            "name": self.name,
            "version": self.version,
            "model_config": self.model_config,
            "config": self.config,
            "metadata": self.metadata,
            "flow_id": self.flow_id,
            "run_id": self.run_id,
            "component_id": self.component_id,
            "parent_id": self.parent_id,
            "prior_component_ids": self.prior_component_ids,
            "intent_executor_ids": self.intent_executor_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "MCPContext":
        """Create context from JSON data"""
        # Convert string component_type back to enum
        if isinstance(data.get("component_type"), str):
            data["component_type"] = ComponentType(data["component_type"])
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate the context configuration"""
        # Base validation
        if not self.name or not self.component_type:
            return False
            
        # Component-specific validation
        if self.component_type == ComponentType.MODEL and not self.model_config:
            return False
            
        return True
        
    def create_child_context(self, component_type: ComponentType, name: str) -> "MCPContext":
        """Create a child context with the same flow and run IDs"""
        child = MCPContext(
            component_type=component_type,
            name=name,
            flow_id=self.flow_id,
            run_id=self.run_id,
            parent_id=self.component_id
        )
        # Add this component as a prior component
        child.prior_component_ids.append(self.component_id)
        return child
        
    def add_prior_component(self, component_id: str) -> None:
        """Add a prior component ID to the list"""
        if component_id not in self.prior_component_ids:
            self.prior_component_ids.append(component_id)
            
    def add_intent_executor(self, executor_id: str) -> None:
        """Add a potential executor ID for this component's intent"""
        if executor_id not in self.intent_executor_ids:
            self.intent_executor_ids.append(executor_id)
            
    def get_execution_chain(self) -> List[str]:
        """Get the chain of component IDs in execution order"""
        return self.prior_component_ids + [self.component_id] 