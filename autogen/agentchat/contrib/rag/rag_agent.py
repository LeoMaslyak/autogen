import re
from termcolor import colored
from typing import Callable, Dict, Optional, Union, List, Tuple, Any

from autogen import logger
from autogen.agentchat import Agent, AssistantAgent, UserProxyAgent
from .datamodel import QueryResults, Query, Document, Chunk
from .prompts import PROMPT_DEFAULT, PROMPT_CODE, PROMPT_QA
from .promptgenerator import PromptGenerator
from .retriever import RetrieverFactory
from .reranker import RerankerFactory
from .encoder import EncoderFactory
from .splitter import SplitterFactory
from .preprocessor import PreProcessor
from .postprocessor import PostProcessor


class RAGAgent(UserProxyAgent):
    def __init__(
        self,
        name="rag_agent",  # default set to RetrieveChatAgent
        human_input_mode: Optional[str] = "ALWAYS",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        retrieve_config: Optional[Dict] = None,  # config for the rag agent
        **kwargs,
    ):
        super().__init__(
            name=name,
            human_input_mode=human_input_mode,
            **kwargs,
        )

        self._retrieve_config = {} if retrieve_config is None else retrieve_config
        self._task = self._retrieve_config.get("task", "default")
        self._retriver = RetrieverFactory.create_retriever(self._retrieve_config.get("retriever", ""))

        # self._client = self._retrieve_config.get("client", chromadb.Client())
        self._docs_path = self._retrieve_config.get("docs_path", None)
        self._extra_docs = self._retrieve_config.get("extra_docs", False)
        self._collection_name = self._retrieve_config.get("collection_name", "autogen-docs")
        if "docs_path" not in self._retrieve_config:
            logger.warning(
                "docs_path is not provided in retrieve_config. "
                f"Will raise ValueError if the collection `{self._collection_name}` doesn't exist. "
                "Set docs_path to None to suppress this warning."
            )
        self._model = self._retrieve_config.get("model", "gpt-4")
        self._max_tokens = self.get_max_tokens(self._model)
        self._chunk_token_size = int(self._retrieve_config.get("chunk_token_size", self._max_tokens * 0.4))
        self._chunk_mode = self._retrieve_config.get("chunk_mode", "multi_lines")
        self._must_break_at_empty_line = self._retrieve_config.get("must_break_at_empty_line", True)
        self._embedding_model = self._retrieve_config.get("embedding_model", "all-MiniLM-L6-v2")
        self._embedding_function = self._retrieve_config.get("embedding_function", None)
        self.customized_prompt = self._retrieve_config.get("customized_prompt", None)
        self.customized_answer_prefix = self._retrieve_config.get("customized_answer_prefix", "").upper()
        self.update_context = self._retrieve_config.get("update_context", True)
        self._get_or_create = self._retrieve_config.get("get_or_create", False) if self._docs_path is not None else True
        # self.custom_token_count_function = self._retrieve_config.get("custom_token_count_function", count_token)
        self.custom_text_split_function = self._retrieve_config.get("custom_text_split_function", None)
        # self._custom_text_types = self._retrieve_config.get("custom_text_types", TEXT_FORMATS)
        self._recursive = self._retrieve_config.get("recursive", True)
        self._context_max_tokens = self._max_tokens * 0.8
        self._collection = True if self._docs_path is None else False  # whether the collection is created
        self._doc_idx = -1  # the index of the current used doc
        self._results = {}  # the results of the current query
        self._intermediate_answers = set()  # the intermediate answers
        self._doc_contents = []  # the contents of the current used doc
        self._doc_ids = []  # the ids of the current used doc
        self._search_string = ""  # the search string used in the current query
        # update the termination message function
        self._is_termination_msg = (
            self._is_termination_msg_retrievechat if is_termination_msg is None else is_termination_msg
        )
        self.register_reply(Agent, RAGAgent._generate_retrieve_user_reply, position=2)

    def _get_prompt(self, question: str, context: str) -> str:
        if self.prompt:
            return self.prompt
        if self.prompt_type == "code":
            return PROMPT_CODE.format(input_question=question, input_context=context)
        if self.prompt_type == "qa":
            return PROMPT_QA.format(input_question=question, input_context=context)
        return PROMPT_DEFAULT.format(input_question=question, input_context=context)

    def _get_retrieval_results(self, query: Query) -> QueryResults:
        return self.retriever.retrieve(query)

    def _get_reranking_results(self, query_results: QueryResults, query: Query) -> QueryResults:
        return self.reranker.rerank(query_results, query)

    def _get_embedding(self, chunk: Chunk) -> Document:
        return self.encoder.encode(chunk)

    def _preprocess(self, question: str, context: str) -> Tuple[str, str]:
        return self.preprocessor.preprocess(question, context)

    def _postprocess(self, answer: str, context: str) -> str:
        return self.postprocessor.postprocess(answer, context)

    def _generate_prompt(self, question: str, context: str) -> str:
        return
