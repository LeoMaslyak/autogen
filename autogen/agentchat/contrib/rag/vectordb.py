from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from datamodel import Document, Query, QueryResults


class VectorDB(ABC):
    """
    Abstract class for vector database. A vector database is responsible for storing and retrieving documents.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def insert_docs(self, docs: List[Document]) -> None:
        """
        Insert documents into the vector database.

        Args:
            docs: A list of documents.
        """
        pass

    @abstractmethod
    def delete_docs(self, ids: List[Any]) -> None:
        """
        Delete documents from the vector database.

        Args:
            ids: A list of document ids.
        """
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


class VectorDBFactory:
    """
    Factory class for creating vector databases.
    """

    @staticmethod
    def create_vector_db(vector_db_name: str, **kwargs) -> VectorDB:
        """
        Create a vector database.

        Args:
            vector_db_name: The name of the vector database.

        Returns:
            A vector database object.
        """
        pass
