"""
symvion.rag.providers
~~~~~~~~~~~~~~~~~~~~~
Pluggable retriever adapters for the Symvion RAG module.

Each provider module implements BaseRetriever and isolates
all provider-specific SDK calls behind the standard search() interface.
Add new providers here without touching Symvion core code.
"""

from symvion.rag.providers.vertex import VertexRetriever

__all__ = ["VertexRetriever"]
