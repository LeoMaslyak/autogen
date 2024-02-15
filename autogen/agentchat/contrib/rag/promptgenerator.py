import re
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union, Callable, Optional, Dict
from autogen.agentchat import AssistantAgent, UserProxyAgent
from .datamodel import Document, Query, QueryResults, Vector, Chunk
from .utils import lazy_import, logger
from .prompts import PROMPT_DEFAULT, PROMPT_CODE, PROMPT_QA, PROMPT_REFINE


def extract_refined_questions(input_text):
    # Define a regular expression pattern to match sentences starting with a number
    pattern = r"\d+\.\s(.*?)(?:\?|\.\.\.)"
    # Use the findall function to extract all matching sentences
    matches = re.findall(pattern, input_text)
    # Remove any leading or trailing whitespace from each match and return the list
    return [match.strip() for match in matches]


class QuestionRefiner:
    """Refine questions using a language model."""

    def __init__(
        self, llm_config: Dict = None, prompt: str = PROMPT_REFINE, post_process_func: Optional[Callable] = None
    ):
        self.assistant = AssistantAgent(
            name="prompt_generator",
            system_message="You are a helpful AI assistant.",
            llm_config=llm_config,
            max_consecutive_auto_reply=1,
            human_input_mode="NEVER",
        )
        self.llm_config = llm_config
        self.prompt = prompt
        self.post_process_func = post_process_func

    def __call__(self, input_question: str, n: int = 3, silent=True) -> str:
        if self.llm_config is None or self.prompt is None:
            logger.warning("LLM config or prompt is not set. Will not refine the input question.")
            return [input_question]
        if "{n}" not in self.prompt or "{input_question}" not in self.prompt:
            raise ValueError("The prompt does not contain '{n}' or '{input_question}'. ")
        message = self.prompt.format(input_question=input_question, n=n)
        self.assistant.reset()
        self.assistant.initiate_chat(self.assistant, message=message, silent=silent)
        self.last_message = self.assistant.last_message().get("content", "")
        self.refined_message = self._post_process(self.last_message)
        return self.refined_message

    def _post_process(self, last_message: str) -> str:
        if self.post_process_func is None:
            return extract_refined_questions(last_message)
        else:
            return self.post_process_func(last_message)
