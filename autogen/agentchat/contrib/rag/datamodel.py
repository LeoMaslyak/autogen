import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Union, Mapping, Sequence
from pydantic.dataclasses import dataclass
from dataclasses import asdict, field


Metadata = Mapping[str, Union[str, int, float, bool, None]]
Vector = Union[Sequence[float], Sequence[int]]
ItemID = Union[str, int]


@dataclass
class Chunk(object):
    """The data sources include texts, codes, database metadata, etc. After processed by splitters, the data sources
    are split into chunks. Chunks are then used to generate documents for the vector database.

    metadata: contains additional information about the document such as source, date, etc.
    """

    title: str
    content: str
    id: ItemID = None
    metadata: Optional[Metadata] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

    def dict(self):
        result = asdict(self)
        return result


@dataclass
class Document(Chunk):
    """A Document is extended from Chunk. It is a record in the vector database."""

    title_embedding: Optional[Vector] = None
    content_embedding: Optional[Vector] = None
    embedding_model: Optional[str] = None
    dimensions: Optional[int] = None


@dataclass
class QueryResults(object):
    ids: List[List[ItemID]]
    texts: Optional[List[List[str]]] = None
    embeddings: Optional[List[List[Vector]]] = None
    metadatas: Optional[List[List[Metadata]]] = None
    distances: Optional[List[List[float]]] = None

    def dict(self):
        result = asdict(self)
        return result


@dataclass
class Query(object):
    """A Query is a request to the vector database for similar documents to the query.

    filters: a dictionary that contains additional conditions for the query.
    """

    text: str
    k: int = 10  # The number of similar documents to return.
    filters: Optional[Dict[str, Any]] = None

    def dict(self):
        result = asdict(self)
        return result
