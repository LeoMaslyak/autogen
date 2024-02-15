import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Union
from .datamodel import Document, Query, QueryResults
from .vectordb import VectorDB, VectorCollection
from .utils import logger

try:
    import chromadb

    if chromadb.__version__ < "0.4.15":
        raise ImportError("Please upgrade chromadb to version 0.4.15 or later.")
    from chromadb.api import ClientAPI as API
    from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT
    from chromadb.api.models.Collection import Collection
except ImportError:
    raise ImportError("Please install chromadb: `pip install chromadb`")

MAX_BATCH_SIZE = os.environ.get("CHROMADB_MAX_BATCH_SIZE", 40000)


class ChromaVectorDB(VectorDB):
    """
    A vector database that uses ChromaDB as the backend.
    """

    def __init__(self, path=None, embedding_function=None, metadata=None, **kwargs):
        """
        # https://github.com/nmslib/hnswlib#supported-distances
        # https://github.com/chroma-core/chroma/blob/566bc80f6c8ee29f7d99b6322654f32183c368c4/chromadb/segment/impl/vector/local_hnsw.py#L184
        # https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        """
        self.path = path
        self.embedding_function = embedding_function
        self.metadata = metadata if metadata else {"hnsw:space": "ip", "hnsw:construction_ef": 30, "hnsw:M": 32}
        if self.path is not None:
            self.client = chromadb.PersistentClient(path=self.path, **kwargs)
        else:
            self.client = chromadb.Client(**kwargs)
        self.active_collection = None

    def create_collection(
        self, collection_name: str, overwrite: bool = False, get_or_create: bool = True
    ) -> Collection:
        try:
            collection = self.clent.get_collection(collection_name)
        except ValueError:
            collection = None
        if collection is None:
            return self.client.create_collection(
                collection_name,
                embedding_function=self.embedding_function,
                get_or_create=get_or_create,
                metadata=self.metadata,
            )
        elif overwrite:
            self.client.delete_collection(collection_name)
            return self.client.create_collection(
                collection_name,
                embedding_function=self.embedding_function,
                get_or_create=get_or_create,
                metadata=self.metadata,
            )
        elif get_or_create:
            return collection
        else:
            raise ValueError(f"Collection {collection_name} already exists.")

    def get_collection(self, collection_name: str = None) -> Collection:
        """
        Get the vector database.
        """
        if collection_name is None:
            if self.active_collection is None:
                raise ValueError("No collection is specified.")
            else:
                logger.info(
                    f"No collection is specified. Using current active collection {self.active_collection.name}."
                )
        else:
            self.active_collection = self.get_collection(collection_name)
        return self.active_collection

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete the vector database.
        """
        self.client.delete_collection(collection_name)
        if self.active_collection.name == collection_name:
            self.active_collection = None

    def _batch_insert(self, collection, embeddings=None, ids=None, metadata=None, documents=None, upsert=False):
        batch_size = int(MAX_BATCH_SIZE)
        for i in range(0, len(documents), min(batch_size, len(documents))):
            end_idx = i + min(batch_size, len(documents) - i)
            if upsert:
                collection.upsert(
                    documents=documents[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    ids=ids[i:end_idx],
                    metadata=metadata[i:end_idx],
                )
            else:
                collection.add(
                    documents=documents[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    ids=ids[i:end_idx],
                    metadata=metadata[i:end_idx],
                )

    def insert_docs(self, docs: List[Document], collection_name: str = None, upsert: bool = False) -> None:
        collection = self.get_collection(collection_name)
        if docs[0].content_embedding is None:
            logger.info(
                "No content embedding is provided. Will use the VectorDB's embedding function to generate the content embedding."
            )
            embeddings = None
        else:
            embeddings = [doc.content_embedding for doc in docs]
        documents = [doc.content for doc in docs]
        ids = [doc.id for doc in docs]
        if docs[0].metadata is None:
            if docs[0].title is not None:
                metadata = [{"title": doc.title} for doc in docs]
            else:
                metadata = None
        else:
            metadata = [doc.metadata for doc in docs]
            if docs[0].title is not None:
                _ = [m.update({"title": doc.title}) for m, doc in zip(metadata, docs)]
        self._batch_insert(collection, embeddings, ids, metadata, documents, upsert)

    def update_docs(self, docs: List[Document], collection_name: str = None) -> None:
        self.insert_docs(docs, collection_name, upsert=True)

    def delete_docs(self, ids: List[Any], collection_name: str = None, **kwargs) -> None:
        collection = self.get_collection(collection_name)
        collection.delete(ids, **kwargs)

    def retrieve_docs(self, queries: List[Query], collection_name: str = None) -> QueryResults:
        collection = self.get_collection(collection_name)
        return collection.query(
            query_texts=[q.text for q in queries],
            n_results=queries[0].k,
            where=queries[0].filter_metadata,
            where_document=queries[0].filter_document,
            include=queries[0].include,
        )
