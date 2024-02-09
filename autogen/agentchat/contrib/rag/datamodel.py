import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Union, Mapping
from pydantic.dataclasses import dataclass
from dataclasses import asdict, field


Metadata = Mapping[str, Union[str, int, float, bool, None]]


@dataclass
class Chunk(object):
    """The data sources include texts, codes, database metadata, etc. After processed by splitters, the data sources
    are split into chunks. Chunks are then used to generate documents for the vector database."""

    title: str
    content: str
    id: Optional[str] = None
    metadata: Optional[Metadata] = None
    """metadata contains additional information about the document such as source, date, etc."""

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

    def dict(self):
        result = asdict(self)
        return result


@dataclass
class Document(Chunk):
    """A Document is extended from Chunk. It is a record in the vector database."""

    title_embedding: Optional[List[float]] = None
    content_embedding: Optional[List[float]] = None


@dataclass
class QueryResults(object):
    ids: List[List[str]]
    texts: Optional[List[List[str]]] = None
    embeddings: Optional[List[List[float]]] = None
    metadatas: Optional[List[List[Metadata]]] = None
    distances: Optional[List[List[float]]] = None

    def dict(self):
        result = asdict(self)
        return result


@dataclass
class Prompt(object):
    type: str  # qa, code, etc.
    prompt: str
    description: Optional[str] = None

    def dict(self):
        result = asdict(self)
        return result
