from typing import Dict, Any, Optional
from protocols.mcp_context import MCPContext, ComponentType

class ToolContext(MCPContext):
    """Context for tool components"""
    
    def __init__(
        self,
        name: str,
        tool_type: str,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            component_type=ComponentType.TOOL,
            name=name,
            config=config or {},
            metadata=metadata or {}
        )
        self.tool_type = tool_type
        
    def validate(self) -> bool:
        """Validate tool configuration"""
        if not super().validate():
            return False
            
        # Tool-specific validation
        if self.tool_type not in ["web_search", "summarizer", "calculator"]:
            return False
            
        # Validate required config based on tool type
        if self.tool_type == "web_search":
            if "max_results" not in self.config:
                return False
        elif self.tool_type == "summarizer":
            if "max_length" not in self.config:
                return False
        elif self.tool_type == "calculator":
            if "precision" not in self.config:
                return False
                
        return True 