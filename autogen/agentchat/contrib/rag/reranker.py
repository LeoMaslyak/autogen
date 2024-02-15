from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from .datamodel import Document, Query, QueryResults
from .utils import lazy_import, logger


class Reranker(ABC):
    """
    Abstract class for reranker. A reranker is responsible for reranking documents based on the query.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def rerank(
        self,
        query: Query,
        docs: Optional[List[Document]] = None,
        return_docs: bool = False,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on the query.

        Args:
            query: A query object.
            docs: A list of documents.

        Returns:
            A list of tuples, each tuple contains a document and its score.
        """
        raise NotImplementedError


class TfidfReranker(Reranker):
    """
    A simple TFIDF reranker.
    """

    def __init__(self):
        TfidfVectorizer = lazy_import("sklearn.feature_extraction.text", "TfidfVectorizer")
        if not TfidfVectorizer:
            raise ImportError("Please install sklearn to use TfidfReranker.")
        self.vectorizer = TfidfVectorizer()
        self.docs_hash = None
        self.docs = None

    def vecotrize(self, docs: List[Document]):
        self.docs = docs
        self.docs_hash = hash(docs)
        self.corpus_tfidf = self.vectorizer.fit_transform([doc.title + "\n" + doc.content for doc in docs])

    def rerank(
        self,
        query: Query,
        docs: Optional[List[Document]] = None,
        return_docs: bool = False,
    ) -> List[Tuple[Document, float]]:
        if docs and self.docs_hash != hash(docs):
            self.vecotrize(docs)
        if not docs and not self.docs:
            raise ValueError("Please provide documents to fit the reranker.")
        query_tfidf = self.vectorizer.transform([query.text])
        scores = self.corpus_tfidf.dot(query_tfidf.T).toarray().flatten()
        ranked_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        if return_docs:
            return [(self.docs[doc[0]], doc[1]) for doc in ranked_docs[: query.k]]
        else:
            return ranked_docs[: query.k]


class RerankerFactory:
    """
    Factory class for creating rerankers.
    """

    @staticmethod
    def create_reranker(reranker_name: str, **kwargs) -> Reranker:
        if reranker_name == "tfidf":
            return TfidfReranker(**kwargs)
        else:
            raise ValueError(f"Reranker {reranker_name} not found")
