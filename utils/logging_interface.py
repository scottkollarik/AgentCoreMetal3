from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class AgentLogger(ABC):
    """Abstract base class for agent logging"""
    
    @abstractmethod
    def start(self):
        """Start logging session"""
        pass
        
    @abstractmethod
    def stop(self):
        """Stop logging session"""
        pass
        
    @abstractmethod
    def log_event(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log a structured event"""
        pass
        
    @abstractmethod
    def log_error(self, error_type: str, message: str, error: Optional[Exception] = None):
        """Log an error"""
        pass
        
    @abstractmethod
    def log_model_response(self, model_name: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a model response"""
        pass
        
    @abstractmethod
    def clear(self):
        """Clear the logs"""
        pass

class NullLogger(AgentLogger):
    """A no-op logger that implements the AgentLogger interface"""
    
    def start(self):
        pass
        
    def stop(self):
        pass
        
    def log_event(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        pass
        
    def log_error(self, error_type: str, message: str, error: Optional[Exception] = None):
        pass
        
    def log_model_response(self, model_name: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        pass
        
    def clear(self):
        pass 