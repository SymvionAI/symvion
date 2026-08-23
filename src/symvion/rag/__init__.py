"""
symvion.rag
~~~~~~~~~~~
Provider-agnostic Retrieval-Augmented Generation (RAG) module.

This module defines the core interfaces and orchestration helpers for RAG
within the Symvion framework. All provider-specific implementations are
located under symvion.rag.providers and all business logic lives
outside this package.
"""

from symvion.rag.base import BaseRetriever
from symvion.rag.pipeline import RAGPipeline

__all__ = ["BaseRetriever", "RAGPipeline"]
