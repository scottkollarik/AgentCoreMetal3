"""Data structures for tracking component state and relationships."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union


class ComponentType(Enum):
    """Types of components in the orchestration system."""

    AGENT = "agent"
    TOOL = "tool"
    MEMORY = "memory"
    MODEL = "model"
    ORCHESTRATOR = "orchestrator"


class LocalModelConfig(TypedDict):
    """Configuration for locally hosted models (e.g. Ollama)."""

    model_name: str
    model_type: str
    temperature: float
    max_tokens: int
    base_url: str


class RemoteModelConfig(TypedDict):
    """Configuration for remote model APIs (e.g. OpenAI)."""

    model_name: str
    api_key: str
    api_base: str
    temperature: float
    max_tokens: int


class ComponentData(TypedDict):
    """Additional runtime information stored on a component."""

    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    state: Dict[str, Any]
    model_config: Union[LocalModelConfig, RemoteModelConfig]
    metrics: Dict[str, float]
    error: Optional[str]


def _generate_short_id() -> str:
    """Generate a short unique identifier."""

    import hashlib
    import uuid

    return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]


@dataclass
class ComponentContext:
    """Context object for managing component state and relationships."""

    # Basic identification
    component_type: ComponentType
    name: str
    version: str = "1.0.0"

    # Flow identifiers
    flow_id: Optional[str] = None
    run_id: Optional[str] = None
    component_id: str = field(default_factory=_generate_short_id)
    parent_id: Optional[str] = None

    # Relationship tracking
    prior_component_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    intent_executor_ids: List[str] = field(default_factory=list)

    # Runtime data
    component_data: ComponentData = field(
        default_factory=lambda: {
            "input_data": {},
            "output_data": {},
            "state": {},
            "model_config": {},
            "metrics": {},
            "error": None,
        }
    )

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:  # pragma: no cover - simple assignment
        if not self.flow_id:
            self.flow_id = _generate_short_id()
        if not self.run_id:
            self.run_id = _generate_short_id()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @property
    def full_id(self) -> str:
        """Return the concatenated flow/run/component identifier."""

        return f"{self.flow_id}:{self.run_id}:{self.component_id}"

    def to_json(self) -> Dict[str, Any]:
        """Serialize the context into a JSON-compatible dictionary."""

        return {
            "component_type": self.component_type.value,
            "name": self.name,
            "version": self.version,
            "flow_id": self.flow_id,
            "run_id": self.run_id,
            "component_id": self.component_id,
            "parent_id": self.parent_id,
            "prior_component_ids": self.prior_component_ids,
            "child_ids": self.child_ids,
            "intent_executor_ids": self.intent_executor_ids,
            "component_data": self.component_data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ComponentContext":
        """Recreate a context from ``to_json`` output."""

        if isinstance(data.get("component_type"), str):
            data["component_type"] = ComponentType(data["component_type"])

        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        return cls(**data)

    # ------------------------------------------------------------------
    # Relationship management
    # ------------------------------------------------------------------
    def create_child_context(
        self, *, component_type: ComponentType, name: str, **kwargs: Any
    ) -> "ComponentContext":
        """Create a child context inheriting flow and run information."""

        child = ComponentContext(
            component_type=component_type,
            name=name,
            version=self.version,
            flow_id=self.flow_id,
            run_id=self.run_id,
            parent_id=self.component_id,
            prior_component_ids=self.prior_component_ids + [self.component_id],
            **kwargs,
        )

        self.child_ids.append(child.component_id)
        return child

    def add_prior_component(self, component_id: str) -> None:
        """Record that another component executed before this one."""

        if component_id not in self.prior_component_ids:
            self.prior_component_ids.append(component_id)

    def add_intent_executor(self, executor_id: str) -> None:
        """Register an executor for this component's intent."""

        if executor_id not in self.intent_executor_ids:
            self.intent_executor_ids.append(executor_id)
            self.updated_at = datetime.utcnow()

    def get_execution_chain(self) -> List[str]:
        """Return IDs for components executed before and including this one."""

        return self.prior_component_ids + [self.component_id]

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def update_component_data(self, **kwargs: Any) -> None:
        """Update tracked data values with validation."""

        for key, value in kwargs.items():
            if key not in self.component_data:
                raise ValueError(f"Invalid data field: {key}")
            self.component_data[key] = value

        self.updated_at = datetime.utcnow()

    def get_component_data(self, key: str) -> Any:
        """Retrieve a value from ``component_data`` with validation."""

        if key not in self.component_data:
            raise ValueError(f"Invalid data field: {key}")
        return self.component_data[key]

