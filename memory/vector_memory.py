from typing import Any, Dict, List, Optional, Callable
from langchain_chroma import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import chromadb

class VectorMemory:
    """Vector-based memory system for storing and retrieving agent history"""

    def __init__(self, config: Dict[str, Any], embedding_fn: Optional[Callable[[str], List[float]]] = None):
        self.config = config
        self.embedding_model = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.max_history = config.get("max_history", 1000)
        self.collection_name = config.get("index_name", "agent_memory")

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
            # Default to Hugging Face embeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        # Initialize vector store
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory="./memory/chroma_db"
        )

    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new memory to the vector store"""
        document = Document(
            page_content=content,
            metadata=metadata or {}
        )
        self.vectorstore.add_documents([document])

    def search_memory(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant memories"""
        return self.vectorstore.similarity_search(query, k=k)

    def get_recent_memories(self, limit: int = 10) -> List[Document]:
        """Get the most recent memories"""
        return self.vectorstore.get()[:limit]

    def clear_memory(self) -> None:
        """Clear all memories"""
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory="./memory/chroma_db"
        )