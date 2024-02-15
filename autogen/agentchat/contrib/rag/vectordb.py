from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from .datamodel import Document, Query, QueryResults


class VectorDB(ABC):
    """
    Abstract class for vector database. A vector database is responsible for storing and retrieving documents.
    """

    def __init__(self, path=None, embedding_function=None, metadata=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def create_collection(self, collection_name: str, overwrite: bool = False, get_or_create: bool = True) -> Any:
        """
        Case 1. if the collection does not exist, create the collection.
        Case 2. the collection exists, if overwrite is True, it will overwrite the collection.
        Case 3. the collection exists and overwrite is False, if get_or_create is True, it will get the collection, otherwise it raise a ValueError.
        """
        raise NotImplementedError

    @abstractmethod
    def get_collection(self, collection_name: str = None) -> Any:
        """
        Get the vector database.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete the vector database.
        """
        raise NotImplementedError

    @abstractmethod
    def insert_docs(self, docs: List[Document], collection_name: str = None, upsert: bool = False) -> None:
        """
        Insert documents into the vector database.

        Args:
            docs: A list of documents.
        """
        raise NotImplementedError

    @abstractmethod
    def update_docs(self, docs: List[Document], collection_name: str = None) -> None:
        """
        Update documents in the vector database.

        Args:
            docs: A list of documents.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_docs(self, ids: List[Any], collection_name: str = None, **kwargs) -> None:
        """
        Delete documents from the vector database.

        Args:
            ids: A list of document ids.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_docs(self, queries: List[Query], collection_name: str = None) -> QueryResults:
        """
        Retrieve documents from the vector database based on the query.

        Args:
            query: A query object.

        Returns:
            A query results object.
        """
        raise NotImplementedError


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
        if vector_db_name == "chromadb":
            from .chromadb import ChromaVectorDB

            return ChromaVectorDB(**kwargs)
