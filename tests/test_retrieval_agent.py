import asyncio
import os
import sys
import json
from unittest.mock import MagicMock, patch

os.environ["OPENAI_API_KEY"] = "sk-placeholder"

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from symvion.core.runtime import Symvion
from symvion.config.models import TenantConfig
from symvion.core.context import TenantContext

async def verify_retrieval():
    print("🧪 Verifying RetrievalAgent Logic...")
    
    # 1. Setup Retrieval Agent for a tenant
    config = TenantConfig(
        tenant_id="test-retrieval",
        enabled_agents=["retrieval"],
        agent_configs={
            "retrieval": {
                "provider": "mock",
                "top_k": 2
            }
        },
        llm_provider="mock"
    )
    runtime = Symvion(config)
    
    # 2. Mock the similarity search
    retrieval_agent = runtime.agents.get_agent("retrieval")
    
    # We need to mock the vectorstore because even the mock provider uses InMemoryVectorStore
    # which might need embeddings initialization.
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "This is a test document about insurance."
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "Symvion offers health and car insurance."
    
    with patch("symvion.vectorstores.factory.VectorStoreFactory.get_vectorstore") as mock_factory:
        mock_vs = MagicMock()
        mock_factory.return_value = mock_vs
        mock_vs.similarity_search.return_value = [mock_doc1, mock_doc2]
        
        # 3. Test execution
        context = TenantContext(tenant_id="test-retrieval")
        result = await retrieval_agent.execute(context, {"message": "Tell me about insurance"})
        
        print(f"Agent Response:\n{result['agent_response']}")
        assert "Found the following relevant information" in result['agent_response']
        assert "insurance" in result['agent_response']
        assert "Symvion" in result['agent_response']
        assert result['metadata']['docs_count'] == 2

    print("\n✅ RetrievalAgent verification passed!")

if __name__ == "__main__":
    asyncio.run(verify_retrieval())
