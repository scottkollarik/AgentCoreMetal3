from typing import Dict, Any, Optional, List
from protocols.mcp_context import MCPContext, ComponentType

class AgentContext(MCPContext):
    """Context for agent components"""
    
    def __init__(
        self,
        name: str,
        agent_type: str,
        model_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            component_type=ComponentType.AGENT,
            name=name,
            model_config=model_config,
            config=config or {},
            metadata=metadata or {}
        )
        self.agent_type = agent_type
        self.tools = tools or []
        
    def validate(self) -> bool:
        """Validate agent configuration"""
        if not super().validate():
            return False
            
        # Agent-specific validation
        if self.agent_type not in ["planning", "execution", "monitoring"]:
            return False
            
        # Validate model config for agents
        if not self.model_config:
            return False
            
        required_model_params = ["model", "temperature"]
        if not all(param in self.model_config for param in required_model_params):
            return False
            
        # Validate tools if specified
        if self.tools and not isinstance(self.tools, list):
            return False
            
        return True 