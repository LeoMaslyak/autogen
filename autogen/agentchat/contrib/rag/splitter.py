from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Callable
from datamodel import Document, Query, QueryResults, Chunk


class Splitter(ABC):
    """
    Abstract class for splitter. A splitter is responsible for splitting raw text into chunks.
    """

    def __init__(
        self,
        max_chunk_length: int = 1024,
        max_chunk_tokens: int = 1024,
        chunk_mode: str = "multi_lines",
        must_break_at_empty_line: bool = True,
        overlap: int = 10,
        encoder: Union[str, Callable] = "default",
    ):
        self.max_chunk_length = max_chunk_length
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_mode = chunk_mode
        self.must_break_at_empty_line = must_break_at_empty_line
        self.overlap = overlap

    @abstractmethod
    def split(
        self,
        input: str,
        max_chunk_length: int = 1024,
        max_chunk_tokens: int = 1024,
        chunk_mode: str = "multi_lines",
    ) -> List[Chunk]:
        """
        Split raw text, code, metadata into chunks.

        Args:
            input: The input text.

        Returns:
            A list of chunks.
        """
        pass


class TextSplitter(Splitter):
    """
    A simple text splitter that splits the input text into chunks based on the delimiter.
    """

    def __init__(self, delimiter: str = "\n", **kwargs):
        super().__init__(**kwargs)
        self.delimiter = delimiter

    def split(self, input: str) -> List[Chunk]:
        chunks = input.split(self.delimiter)
        return [Chunk(content=chunk) for chunk in chunks]
