from typing import Dict, Any, Optional
from protocols.mcp_context import MCPContext, ComponentType

class MemoryContext(MCPContext):
    """Context for memory components"""
    
    def __init__(
        self,
        name: str,
        memory_type: str = "vector",
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            component_type=ComponentType.MEMORY,
            name=name,
            config=config or {},
            metadata=metadata or {}
        )
        self.memory_type = memory_type
        
    def validate(self) -> bool:
        """Validate memory configuration"""
        if not super().validate():
            return False
            
        # Memory-specific validation
        if self.memory_type not in ["vector", "sql", "redis"]:
            return False
            
        # Validate required config based on memory type
        if self.memory_type == "vector":
            if "embedding_model" not in self.config:
                return False
        elif self.memory_type == "sql":
            if "connection_string" not in self.config:
                return False
        elif self.memory_type == "redis":
            if "host" not in self.config or "port" not in self.config:
                return False
                
        return True 