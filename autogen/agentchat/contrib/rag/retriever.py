from abc import ABC, abstractmethod
from typing import List, Tuple
from datamodel import Document, Query, QueryResults


class Retriever(ABC):
    """
    Abstract class for retriever. A retriever is responsible for retrieving documents from the vector database.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def retrieve_docs(self, query: Query) -> QueryResults:
        """
        Retrieve documents from the vector database based on the query.

        Args:
            query: A query object.

        Returns:
            A query results object.
        """
        pass


class RetrieverFactory:
    """
    Factory class for creating retrievers.
    """

    @staticmethod
    def create_retriever(retriever_name: str, **kwargs) -> Retriever:
        """
        Create a retriever.

        Args:
            retriever_name: The name of the retriever.

        Returns:
            A retriever object.
        """
        pass
