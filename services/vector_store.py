import logging
from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStoreService:
    """Service for managing vector store operations."""
    
    def __init__(self, config_path: str = "configs/vector_store_config.yaml"):
        """Initialize the vector store service.
        
        Args:
            config_path: Path to vector store configuration
        """
        self.config = self._load_config(config_path)
        self.client = self._initialize_client()
        self.collections: Dict[str, chromadb.Collection] = {}
        self._initialize_collections()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load vector store configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)["vector_store"]
    
    def _initialize_client(self) -> chromadb.Client:
        """Initialize ChromaDB client."""
        settings = Settings(
            chroma_server_host=os.getenv("CHROMA_SERVER_HOST", "localhost"),
            chroma_server_port=int(os.getenv("CHROMA_SERVER_PORT", "8000")),
            allow_reset=self.config["settings"]["allow_reset"],
            anonymized_telemetry=self.config["settings"]["anonymized_telemetry"]
        )
        return chromadb.Client(settings)
    
    def _initialize_collections(self):
        """Initialize configured collections."""
        for collection_name, collection_config in self.config["collections"].items():
            try:
                # Get or create collection
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self._get_embedding_function(collection_config)
                )
                self.collections[collection_name] = collection
                logger.info(f"Initialized collection: {collection_name}")
            except Exception as e:
                logger.error(f"Error initializing collection {collection_name}: {e}")
    
    def _get_embedding_function(self, config: Dict[str, Any]) -> Any:
        """Get embedding function based on configuration."""
        if config["embedding_function"] == "sentence-transformers":
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config["embedding_model"]
            )
        raise ValueError(f"Unsupported embedding function: {config['embedding_function']}")
    
    async def add_memory(
        self,
        collection: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add memory entries to a collection."""
        if collection not in self.collections:
            raise ValueError(f"Collection {collection} not found")
            
        try:
            self.collections[collection].add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(texts)} entries to collection {collection}")
        except Exception as e:
            logger.error(f"Error adding memory to {collection}: {e}")
            raise
    
    async def query_memory(
        self,
        collection: str,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query memory entries from a collection."""
        if collection not in self.collections:
            raise ValueError(f"Collection {collection} not found")
            
        try:
            results = self.collections[collection].query(
                query_texts=query_texts,
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            logger.error(f"Error querying memory from {collection}: {e}")
            raise
    
    async def delete_memory(
        self,
        collection: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ):
        """Delete memory entries from a collection."""
        if collection not in self.collections:
            raise ValueError(f"Collection {collection} not found")
            
        try:
            self.collections[collection].delete(
                ids=ids,
                where=where
            )
            logger.info(f"Deleted entries from collection {collection}")
        except Exception as e:
            logger.error(f"Error deleting memory from {collection}: {e}")
            raise
    
    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        if collection not in self.collections:
            raise ValueError(f"Collection {collection} not found")
            
        try:
            count = self.collections[collection].count()
            return {
                "collection": collection,
                "count": count,
                "embedding_function": self.collections[collection]._embedding_function.__class__.__name__
            }
        except Exception as e:
            logger.error(f"Error getting stats for {collection}: {e}")
            raise 