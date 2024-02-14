from abc import ABC, abstractmethod
from typing import List, Tuple
from datamodel import Document, Query, QueryResults


class Reranker(ABC):
    """
    Abstract class for reranker. A reranker is responsible for reranking documents based on the query.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def rerank_docs(self, query: Query, docs: List[Document]) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on the query.

        Args:
            query: A query object.
            docs: A list of documents.

        Returns:
            A list of tuples, each tuple contains a document and its score.
        """
        raise NotImplementedError


class BM25Reranker(Reranker):
    """
    A simple BM25 reranker.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def rerank_docs(self, query: Query, docs: List[Document]) -> List[Tuple[Document, float]]:
        return [(doc, 1.0) for doc in docs]


class EmbeddingReranker(Reranker):
    """
    An embedding reranker.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def rerank_docs(self, query: Query, docs: List[Document]) -> List[Tuple[Document, float]]:
        return [(doc, 1.0) for doc in docs]


class RerankerFactory:
    """
    Factory class for creating rerankers.
    """

    @staticmethod
    def create_reranker(reranker_name: str, **kwargs) -> Reranker:
        if reranker_name == "bm25":
            return BM25Reranker(**kwargs)
        elif reranker_name == "embedding":
            return EmbeddingReranker(**kwargs)
        else:
            raise ValueError(f"Reranker {reranker_name} not found")
