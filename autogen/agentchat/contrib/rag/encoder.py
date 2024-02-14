from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Callable
from .datamodel import Document, Query, QueryResults, Vector, Chunk
from .utils import lazy_import


class EmbeddingFunction(ABC):
    """
    Abstract class for embedding function. An embedding function is responsible for embedding text, images, etc. into vectors.

    """

    model_name: str = None
    dimensions: int = None

    @abstractmethod
    def __call__(self, inputs: Union[str, List[str]]) -> List[Vector]:
        """
        Embed inputs into vectors.

        Args:
            inputs: A list of inputs. If the input is a string, it will be converted to a list of length 1.

        Returns:
            A list of vectors.
        """
        raise NotImplementedError


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """More models can be found at https://www.sbert.net/docs/pretrained_models.html."""

    def __init__(self, model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        self.model_name = model_name
        self.sentence_transformer = lazy_import("sentence_transformers", "SentenceTransformer")(model_name)
        if not self.sentence_transformer:
            raise ImportError("Please install sentence_transformers to use SentenceTransformerEmbeddingFunction.")
        self.dimensions = self.sentence_transformer.encode(["dummy"]).shape[1]

    def __call__(self, inputs: Union[str, List[str]]) -> List[Vector]:
        if isinstance(inputs, str):
            inputs = [inputs]
        return self.sentence_transformer.encode(inputs).tolist()


class Encoder:
    """
    An encoder is responsible for encoding text, images, etc. into vectors.

    """

    def __init__(self, embedding_function: EmbeddingFunction = SentenceTransformerEmbeddingFunction()):
        self._embedding_function = embedding_function

    def encode_chunks(self, chunks: List[Chunk]) -> List[Document]:
        """
        Encode Chunks into Documents.

        Args:
            chunks: A list of Chunk.

        Returns:
            A list of Document.
        """
        return [
            Document(
                **chunk.dict(),
                title_embedding=self._embedding_function(chunk.title)[0] if chunk.title else [],
                content_embedding=self.embedding_function(chunk.content)[0] if chunk.content else [],
                embedding_model=self._embedding_function.model_name,
                dimensions=self._embedding_function.dimensions
            )
            for chunk in chunks
        ]

    @property
    def embedding_function(self) -> EmbeddingFunction:
        """
        Get the embedding function.

        Returns:
            The embedding function.
        """
        return self._embedding_function

    @embedding_function.setter
    def embedding_function(self, new_embedding_function: EmbeddingFunction) -> None:
        """
        Set the embedding function.

        Args:
            new_embedding_function: The new embedding function.
        """
        self._embedding_function = new_embedding_function
