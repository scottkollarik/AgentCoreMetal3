from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from memory.vector_memory import VectorMemory
from memory.error_memory import ErrorMemory

class MemoryProvider(ABC):
    """Abstract base class for memory providers"""
    
    @abstractmethod
    async def store(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store data in memory"""
        pass
        
    @abstractmethod
    async def retrieve(self, query: str, limit: int = 5) -> List[Any]:
        """Retrieve data from memory"""
        pass
        
    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored data"""
        pass

class VectorMemoryProvider(MemoryProvider):
    """Provider for vector-based memory storage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.memory = VectorMemory(config)
        
    async def store(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store data in vector memory"""
        try:
            return await self.memory.store(data, metadata)
        except Exception as e:
            raise ValueError(f"Failed to store in vector memory: {str(e)}")
            
    async def retrieve(self, query: str, limit: int = 5) -> List[Any]:
        """Retrieve data from vector memory"""
        try:
            return await self.memory.retrieve(query, limit)
        except Exception as e:
            raise ValueError(f"Failed to retrieve from vector memory: {str(e)}")
            
    async def clear(self) -> None:
        """Clear vector memory"""
        try:
            await self.memory.clear()
        except Exception as e:
            raise ValueError(f"Failed to clear vector memory: {str(e)}")

class ErrorMemoryProvider(MemoryProvider):
    """Provider for error-based memory storage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.memory = ErrorMemory(config)
        
    async def store(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store error in memory"""
        try:
            return await self.memory.add_error(data, metadata)
        except Exception as e:
            raise ValueError(f"Failed to store in error memory: {str(e)}")
            
    async def retrieve(self, query: str, limit: int = 5) -> List[Any]:
        """Retrieve errors from memory"""
        try:
            return await self.memory.get_errors(query, limit)
        except Exception as e:
            raise ValueError(f"Failed to retrieve from error memory: {str(e)}")
            
    async def clear(self) -> None:
        """Clear error memory"""
        try:
            await self.memory.clear()
        except Exception as e:
            raise ValueError(f"Failed to clear error memory: {str(e)}")
            
    async def get_error_patterns(self) -> List[Dict[str, Any]]:
        """Get error patterns from memory"""
        try:
            return await self.memory.get_error_patterns()
        except Exception as e:
            raise ValueError(f"Failed to get error patterns: {str(e)}") 