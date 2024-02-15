from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union, Callable, Optional, Dict
from autogen.agentchat import AssistantAgent, UserProxyAgent
from .datamodel import Document, Query, QueryResults, Vector, Chunk
from .utils import lazy_import, logger
from .prompts import PROMPT_DEFAULT, PROMPT_CODE, PROMPT_QA, PROMPT_REFINE


def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


@singleton
class PromptGenerator:
    """Refine a prompt with the given context and question."""

    def __init__(self, prompt: str = PROMPT_REFINE, llm_config: Optional[Dict] = None):
        self.assistant = AssistantAgent(
            name="prompt_generator",
            system_message="You are a helpful AI assistant.",
            llm_config=llm_config,
            max_consecutive_auto_reply=1,
            human_input_mode="NEVER",
        )
        self.prompt = prompt

    def __call__(self, input: str) -> str:
        self.assistant.reset()
        self.assistant.initiate_chat(self.assistant)
