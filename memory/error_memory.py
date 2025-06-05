from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import logging
from .vector_memory import VectorMemory
from utils.config_validator import ConfigValidator, ErrorMemoryConfig

logger = logging.getLogger(__name__)

class ErrorMemory:
    """Memory system for storing and retrieving error information"""
    
    def __init__(self, config: Dict[str, Any], embedding_fn: Optional[Callable[[str], List[float]]] = None):
        """Initialize error memory with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - use_vector_store: Whether to use vector store for errors
                - embedding_model: Model name for embeddings
                - max_history: Maximum number of items to store
                - index_name: Name of the collection
                - base_url: Base URL for embedding service
                - persist_directory: Directory for persisting data
            embedding_fn: Optional custom embedding function
        """
        # Validate configuration
        self.config = ConfigValidator.validate_error_memory_config(config)
        
        # Extract validated configuration
        self.use_vector_store = self.config.use_vector_store
        self.max_history = self.config.max_history
        
        if self.use_vector_store:
            try:
                # Configure vector store for errors
                vector_config = {
                    "type": "vector",
                    "embedding_model": self.config.embedding_model,
                    "max_history": self.config.max_history,
                    "index_name": self.config.index_name,
                    "base_url": self.config.base_url,
                    "persist_directory": self.config.persist_directory
                }
                self.vector_memory = VectorMemory(vector_config, embedding_fn)
            except Exception as e:
                logger.error(f"Failed to initialize vector store for error memory: {str(e)}")
                self.use_vector_store = False
                self.vector_memory = None
        
    def log_error(self, error_message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an error with context"""
        # Always log to Python's logging system
        logger.error(
            error_message,
            extra={
                "error_type": context.get("error_type", "Unknown") if context else "Unknown",
                "component": context.get("component", "Unknown") if context else "Unknown",
                **(context or {})
            }
        )
        
        # If vector store is available, store the error there too
        if self.use_vector_store and self.vector_memory:
            try:
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "error_type": context.get("error_type", "Unknown") if context else "Unknown",
                    "component": context.get("component", "Unknown") if context else "Unknown"
                }
                if context:
                    metadata.update(context)
                    
                self.vector_memory.add_memory(error_message, metadata)
            except Exception as e:
                logger.error(f"Failed to store error in vector memory: {str(e)}")
        
    def get_error_patterns(self) -> Dict[str, int]:
        """Get patterns of errors"""
        patterns = {}
        
        if self.use_vector_store and self.vector_memory:
            try:
                errors = self.vector_memory.search_memory("", k=self.vector_memory.max_history)
                for error in errors:
                    error_type = error.metadata.get("error_type", "Unknown")
                    patterns[error_type] = patterns.get(error_type, 0) + 1
            except Exception as e:
                logger.error(f"Failed to get error patterns from vector memory: {str(e)}")
                
        return patterns 