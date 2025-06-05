from typing import Any, Dict, List, Optional, Callable
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import chromadb
import warnings
import logging
from utils.config_validator import ConfigValidator, MemoryConfig

logger = logging.getLogger(__name__)

class VectorMemory:
    """Vector-based memory system for storing and retrieving agent history"""

    def __init__(self, config: Dict[str, Any], embedding_fn: Optional[Callable[[str], List[float]]] = None):
        """Initialize vector memory with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - embedding_model: Model name for embeddings
                - max_history: Maximum number of items to store
                - index_name: Name of the collection
                - base_url: Base URL for embedding service
                - persist_directory: Directory for persisting data
            embedding_fn: Optional custom embedding function
        """
        # Validate configuration
        self.config = ConfigValidator.validate_memory_config(config)
        
        # Extract validated configuration
        self.embedding_model = self.config.embedding_model
        self.max_history = self.config.max_history
        self.collection_name = self.config.index_name
        self.base_url = self.config.base_url
        self.persist_directory = self.config.persist_directory

        if embedding_fn is not None:
            # Wrap the embedding_fn so it can handle a list of texts (as Chroma expects)
            class EmbeddingWrapper:
                def __init__(self, fn):
                    self.fn = fn
                def __call__(self, texts):
                    # Chroma expects a list of texts, so map the function
                    return [self.fn(text) for text in texts]
                def embed_documents(self, texts):
                    # This is what Chroma calls internally
                    return [self.fn(text) for text in texts]
                def embed_query(self, text):
                    # For single text queries
                    return self.fn(text)
            self.embeddings = EmbeddingWrapper(embedding_fn)
        else:
            # Use configured embedding model
            with warnings.catch_warnings(record=True) as w:
                self.embeddings = OllamaEmbeddings(
                    model=self.embedding_model,
                    base_url=self.base_url
                )
                # Log any deprecation warnings
                for warning in w:
                    if issubclass(warning.category, DeprecationWarning):
                        logger.warning(
                            f"Deprecation warning in VectorMemory: {warning.message}",
                            extra={
                                "component": "VectorMemory",
                                "model": self.embedding_model,
                                "base_url": self.base_url
                            }
                        )

        try:
            # Initialize vector store
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize vector store: {str(e)}",
                extra={
                    "component": "VectorMemory",
                    "collection": self.collection_name,
                    "model": self.embedding_model
                }
            )
            raise

    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new memory to the vector store"""
        try:
            document = Document(
                page_content=content,
                metadata=metadata or {}
            )
            self.vectorstore.add_documents([document])
        except Exception as e:
            logger.error(
                f"Failed to add memory: {str(e)}",
                extra={
                    "component": "VectorMemory",
                    "content_length": len(content),
                    "metadata_keys": list(metadata.keys()) if metadata else []
                }
            )
            raise

    def search_memory(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant memories"""
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(
                f"Failed to search memory: {str(e)}",
                extra={
                    "component": "VectorMemory",
                    "query_length": len(query),
                    "k": k
                }
            )
            raise

    def get_recent_memories(self, limit: int = 10) -> List[Document]:
        """Get the most recent memories"""
        try:
            return self.vectorstore.get()[:limit]
        except Exception as e:
            logger.error(
                f"Failed to get recent memories: {str(e)}",
                extra={
                    "component": "VectorMemory",
                    "limit": limit
                }
            )
            raise

    def clear_memory(self) -> None:
        """Clear all memories"""
        try:
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            logger.error(
                f"Failed to clear memory: {str(e)}",
                extra={
                    "component": "VectorMemory",
                    "collection": self.collection_name
                }
            )
            raise