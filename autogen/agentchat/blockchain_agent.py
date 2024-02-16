# autogen/agentchat/assistant_agent.py

from typing import Callable, Dict, Literal, Optional, Union
import os

from .conversable_agent import ConversableAgent
from autogen.runtime_logging import logging_enabled, log_new_agent
from autogen.oai_wrapper import OpenAIWrapper  # Ensure this import matches your project structure

class AssistantAgent(ConversableAgent):
    """Assistant agent specialized in explaining blockchain terminology."""

    DEFAULT_SYSTEM_MESSAGE = """You are a knowledgeable assistant specialized in blockchain technology.
Your task is to explain blockchain terminology and concepts to the user in a clear and accessible manner.
For each user query related to blockchain, provide a detailed explanation that includes the term's definition,
its significance in the blockchain ecosystem, and any relevant examples or use cases.
Aim to make your explanations helpful for both beginners and those with intermediate knowledge of blockchain.
Remember to keep your language simple and avoid technical jargon unless you are defining it.
Reply "TERMINATE" in the end when everything is done."""

    DEFAULT_DESCRIPTION = "A specialized AI assistant focused on explaining blockchain terminology and concepts."

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = None,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        description: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name,
            system_message or self.DEFAULT_SYSTEM_MESSAGE,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            llm_config=llm_config,
            description=description or self.DEFAULT_DESCRIPTION,
            **kwargs,
        )
        if logging_enabled():
            log_new_agent(self, locals())

        self.openai_wrapper = OpenAIWrapper()

    def handle_user_input(self, input_message: str) -> Dict:
        """Process user input related to blockchain and generate a relevant explanation."""
        try:
            response = self.openai_wrapper.generate_response(f"Explain the blockchain term: {input_message}")
            return {"response": response}
        except Exception as e:
            print(f"Error processing blockchain query: {e}")
            return {"error": "An error occurred while processing your blockchain query."}

# Example usage
if __name__ == "__main__":
    assistant_agent = AssistantAgent(name="Blockchain Explainer Agent", llm_config=False)
    blockchain_query = "What is a smart contract?"
    explanation = assistant_agent.handle_user_input(blockchain_query)
    print(explanation)
