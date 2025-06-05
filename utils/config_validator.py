from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
import yaml
from pathlib import Path

class MemoryConfig(BaseModel):
    """Memory configuration schema"""
    type: str = Field(..., description="Type of memory system (vector, etc)")
    embedding_model: str = Field(..., description="Model name for embeddings")
    max_history: int = Field(..., gt=0, description="Maximum number of items to store")
    index_name: str = Field(..., description="Name of the collection")
    base_url: str = Field(..., description="Base URL for embedding service")
    persist_directory: str = Field(..., description="Directory for persisting data")

    @validator('type')
    def validate_type(cls, v):
        allowed_types = ['vector']
        if v not in allowed_types:
            raise ValueError(f"Memory type must be one of {allowed_types}")
        return v

class ErrorMemoryConfig(BaseModel):
    """Error memory configuration schema"""
    use_vector_store: bool = Field(True, description="Whether to use vector store for errors")
    embedding_model: str = Field(..., description="Model name for embeddings")
    max_history: int = Field(..., gt=0, description="Maximum number of items to store")
    index_name: str = Field(..., description="Name of the collection")
    base_url: str = Field(..., description="Base URL for embedding service")
    persist_directory: str = Field(..., description="Directory for persisting data")

class AgentConfig(BaseModel):
    """Agent configuration schema"""
    model_name: str = Field(..., description="Name of the model to use")
    model_type: str = Field(..., description="Type of model (ollama, openai, etc)")
    temperature: float = Field(..., ge=0.0, le=1.0, description="Model temperature")
    max_tokens: int = Field(..., gt=0, description="Maximum tokens to generate")
    system_prompt: str = Field(..., description="System prompt for the agent")

    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_types = ['ollama', 'openai']
        if v not in allowed_types:
            raise ValueError(f"Model type must be one of {allowed_types}")
        return v

class ToolConfig(BaseModel):
    """Tool configuration schema"""
    enabled: bool = Field(..., description="Whether the tool is enabled")
    max_results: Optional[int] = Field(None, gt=0, description="Maximum results to return")
    max_length: Optional[int] = Field(None, gt=0, description="Maximum length for operations")

class ConfigValidator:
    """Validates configuration files and direct configurations"""
    
    @staticmethod
    def validate_memory_config(config: Dict[str, Any]) -> MemoryConfig:
        """Validate memory configuration"""
        return MemoryConfig(**config)
    
    @staticmethod
    def validate_error_memory_config(config: Dict[str, Any]) -> ErrorMemoryConfig:
        """Validate error memory configuration"""
        return ErrorMemoryConfig(**config)
    
    @staticmethod
    def validate_agent_config(config: Dict[str, Any]) -> AgentConfig:
        """Validate agent configuration"""
        return AgentConfig(**config)
    
    @staticmethod
    def validate_tool_config(config: Dict[str, Any]) -> ToolConfig:
        """Validate tool configuration"""
        return ToolConfig(**config)
    
    @staticmethod
    def load_and_validate_yaml(config_path: str) -> Dict[str, Any]:
        """Load and validate YAML configuration file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate each section
        validated_config = {}
        
        # Validate memory configs
        if 'memory' in config:
            validated_config['memory'] = ConfigValidator.validate_memory_config(config['memory'])
        if 'error_memory' in config:
            validated_config['error_memory'] = ConfigValidator.validate_error_memory_config(config['error_memory'])
            
        # Validate agent configs
        for agent_type in ['planning_agent', 'execution_agent', 'monitoring_agent']:
            if agent_type in config:
                validated_config[agent_type] = ConfigValidator.validate_agent_config(config[agent_type])
                
        # Validate tool configs
        if 'tools' in config:
            validated_config['tools'] = {
                tool_name: ConfigValidator.validate_tool_config(tool_config)
                for tool_name, tool_config in config['tools'].items()
            }
            
        return validated_config 