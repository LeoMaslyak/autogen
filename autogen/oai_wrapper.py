# autogen/oai_wrapper.py

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class OpenAIWrapper:
    def __init__(self):
        pass

    def generate_response(self, prompt):
        try:
            response = client.chat.completions.create(
                model="gpt-4",  # Adjust the model as necessary
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return "I'm sorry, but I was unable to generate a response."
