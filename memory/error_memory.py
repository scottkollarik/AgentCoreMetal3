from typing import Any, Dict, List, Optional, Callable
from .vector_memory import VectorMemory
from datetime import datetime

class ErrorMemory(VectorMemory):
    """Vector-based memory system specifically for storing and analyzing errors"""

    def __init__(self, config: Dict[str, Any], embedding_fn: Optional[Callable[[str], List[float]]] = None):
        # Override the collection name to separate error storage
        config["index_name"] = "error_memory"
        super().__init__(config, embedding_fn)

    def add_error(self, error: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Add an error to the vector store with metadata"""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            **(context or {})
        }
        self.add_memory(error, metadata)

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent errors with their metadata"""
        return self.get_recent_memories(limit)

    def search_similar_errors(self, error: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar errors that occurred in the past"""
        return self.search_memory(error, k=k)

    def get_error_patterns(self) -> Dict[str, int]:
        """Analyze error patterns and return frequency counts"""
        errors = self.get_recent_memories(limit=1000)  # Get a larger sample
        patterns = {}
        for error in errors:
            error_type = error.metadata.get("error_type", "unknown")
            patterns[error_type] = patterns.get(error_type, 0) + 1
        return patterns 