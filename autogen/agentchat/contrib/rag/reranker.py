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
        pass


class RerankerFactory:
    """
    Factory class for creating rerankers.
    """

    @staticmethod
    def create_reranker(reranker_name: str, **kwargs) -> Reranker:
        """
        Create a reranker.

        Args:
            reranker_name: The name of the reranker.

        Returns:
            A reranker object.
        """
        pass
