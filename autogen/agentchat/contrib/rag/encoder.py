from abc import ABC, abstractmethod
from typing import List, Tuple
from datamodel import Document, Query, QueryResults, Vector


class Encoder(ABC):
    """
    Abstract class for encoder. An encoder is responsible for encoding documents into vectors.
    """

    def __init__(self, **kwargs):
        self._embedding_function = kwargs.get("embedding_function", None)
        pass

    @abstractmethod
    def encode_docs(self, docs: List[Document]) -> List[Tuple[Document, Vector]]:
        """
        Encode documents into vectors.

        Args:
            docs: A list of documents.

        Returns:
            A list of tuples, each tuple contains a document and its vector.
        """
        pass

    @abstractmethod
    def embedding_function(self) -> Vector:
        """
        Get the embedding function.

        Returns:
            The embedding function.
        """
        pass
