from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from .datamodel import Document, Query, QueryResults, Vector, Chunk
from .encoder import Encoder, EmbeddingFunction
from .vectordb import VectorDB


class Retriever(ABC):
    """
    Abstract class for retriever. A retriever is responsible for retrieving documents from the vector database.
    """

    def __init__(
        self,
        db_type: str,
        db_config: Dict,
        encoder: Encoder,
    ) -> VectorDB:
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
        raise NotImplementedError

    def insert_docs(self, docs: List[Document]) -> None:
        """
        Insert documents into the vector database.

        Args:
            docs: A list of documents.
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
