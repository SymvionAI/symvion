import asyncio
from typing import Dict, Any, Optional, List
from symvion.agents.base import BaseAgent
from symvion.core.context import TenantContext
from symvion.core.logger import logger
from symvion.vectorstores.factory import VectorStoreFactory
from langchain_openai import OpenAIEmbeddings

class RetrievalAgent(BaseAgent):
    """
    A generic retrieval agent for semantic search across various vector stores.
    """
    def __init__(self, tenant_id: str, config: Dict[str, Any]):
        super().__init__(tenant_id, config)
        self.provider = config.get("provider", "mock")
        self.vs_config = config.get("vectorstore_config", {})
        self.embeddings = OpenAIEmbeddings() # Default embeddings
        self._vectorstore = None

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = VectorStoreFactory.get_vectorstore(
                self.provider, self.vs_config, self.embeddings
            )
        return self._vectorstore

    async def execute(self, context: TenantContext, input_data: Dict[str, Any], tools: Optional[Any] = None) -> Dict[str, Any]:
        query = input_data.get("message", "")
        if not query:
            return {"agent_response": "Please provide a query for retrieval.", "token_usage": {}}

        logger.info("RETRIEVAL_START", context, provider=self.provider, query=query)
        
        # Perform similarity search
        docs = await asyncio.to_thread(self.vectorstore.similarity_search, query, k=self.config.get("top_k", 3))
        
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        if not context_text:
            return {
                "agent_response": "I couldn't find any relevant information for your query.",
                "token_usage": {"input_tokens": 0, "output_tokens": 0}
            }

        return {
            "agent_response": f"Found the following relevant information:\n\n{context_text}",
            "token_usage": {"input_tokens": 0, "output_tokens": 0}, # Retrieval usually doesn't consume LLM tokens unless using a generator
            "metadata": {"docs_count": len(docs)}
        }
