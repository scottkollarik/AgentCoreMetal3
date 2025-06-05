from typing import Dict, List, Any, Optional, Protocol
from langchain.schema import BaseMessage, AIMessage, HumanMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun

class ModelContextProtocol(Protocol):
    """Protocol defining how models interact with their context"""
    
    def format_messages(self, 
                       messages: List[BaseMessage], 
                       **kwargs: Any) -> List[BaseMessage]:
        """Format messages for the model's context window"""
        ...
    
    def count_tokens(self, 
                    messages: List[BaseMessage]) -> int:
        """Count tokens in the messages"""
        ...
    
    def truncate_context(self, 
                        messages: List[BaseMessage], 
                        max_tokens: int) -> List[BaseMessage]:
        """Truncate context to fit within token limit"""
        ...
    
    def parse_response(self, 
                      response: str, 
                      **kwargs: Any) -> Dict[str, Any]:
        """Parse and validate model response"""
        ...
    
    def create_prompt_template(self, 
                             template: str, 
                             **kwargs: Any) -> str:
        """Create a model-specific prompt template"""
        ...
    
    def handle_streaming(self, 
                        response: Any, 
                        callback_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        """Handle streaming responses from the model"""
        ...
    
    def validate_model_params(self, 
                            params: Dict[str, Any]) -> bool:
        """Validate model parameters"""
        ... 