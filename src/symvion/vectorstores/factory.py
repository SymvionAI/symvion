from typing import Dict, Any, Optional
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from symvion.providers.factory import ProviderFactory

class VectorStoreFactory:
    """
    Factory to initialize LangChain vector stores.
    Supports Pinecone, Qdrant, PGVector, and a Mock store.
    """
    
    @staticmethod
    def get_vectorstore(provider: str, config: Dict[str, Any], embeddings: Embeddings) -> VectorStore:
        provider = provider.lower()
        
        if provider == "mock":
            from langchain_core.vectorstores import InMemoryVectorStore
            return InMemoryVectorStore(embedding=embeddings)
            
        if provider == "pinecone":
            from langchain_pinecone import PineconeVectorStore
            return PineconeVectorStore(
                index_name=config["index_name"],
                embedding=embeddings,
                pinecone_api_key=config.get("api_key")
            )
            
        if provider == "qdrant":
            from langchain_qdrant import QdrantVectorStore
            return QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name=config["collection_name"],
                url=config.get("url"),
                api_key=config.get("api_key")
            )
            
        if provider == "pgvector":
            from langchain_postgres import PGVector
            return PGVector(
                connection=config["connection_string"],
                embeddings=embeddings,
                collection_name=config.get("collection_name", "langchain")
            )
            
        raise ValueError(f"Unsupported vector store provider: {provider}")
