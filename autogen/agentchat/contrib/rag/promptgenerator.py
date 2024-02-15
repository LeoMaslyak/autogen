from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Callable, Optional, Dict
from autogen.agentchat import AssistantAgent
from .datamodel import Document, Query, QueryResults, Vector, Chunk
from .utils import lazy_import, logger
from .prompts import PROMPT_DEFAULT, PROMPT_CODE, PROMPT_QA, PROMPT_REFINE


class PromptGenerator:
    """Refine a prompt with the given context and question."""

    def __init__(self, prompt: str = PROMPT_REFINE, llm_config: Optional[Dict] = None):
        self.prompt = prompt
