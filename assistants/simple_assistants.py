import os

# Add the parent directory to sys.path
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict, Tuple

from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_gigachat import GigaChat
#from yandex_chain import YandexLLM
from langchain_community.llms import YandexGPT
#from langchain_community.llms import YandexGPT


from abc import abstractmethod
from typing import List, Any, Optional, Dict, Tuple

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAssistant:
    def __init__(self, system_prompt: Optional[str] = None):
        logger.info("Initializing chat model")
        self.llm = self.initialize()
        self.conversation_history = []
        if system_prompt:
            self.set_system_prompt(system_prompt)
        else:
            logger.warning("No system prompt provided. The assistant may not behave as expected.")

        logger.info("Initialized")

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt


    @abstractmethod
    def initialize(self):
        """
            Initialize model here.
        """

    def ask_question(self, query: str) -> str:
        if self.llm is None:
            logger.error("RAG chain not initialized")
            raise ValueError("Model or RAG chain not initialized.")
        try:
            result = self.llm.invoke(query)
            return result.content
        except AttributeError as e:
            logger.error(f"AttributeError in ask_question: {str(e)}")
            raise ValueError(f"Error processing query: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in ask_question: {str(e)}")
            raise


class SimpleAssistantGPT(SimpleAssistant):
    def __init__(self, system_prompt):
        super().__init__(system_prompt)
    def initialize(self):
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4)


class SimpleAssistantMistralAI(SimpleAssistant):
    def __init__(self, system_prompt):
        super().__init__(system_prompt)
    def initialize(self):
        return ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.4)


class SimpleAssistantYA(SimpleAssistant):
    def __init__(self, system_prompt):
        super().__init__(system_prompt)
    def initialize(self):
        return YandexGPT(
            api_key = config.YA_API_KEY, 
            folder_id=config.YA_FOLDER_ID, 
            model_uri=f'gpt://{config.YA_FOLDER_ID}/yandexgpt/latest',
            temperature=0.4
            )

class SimpleAssistantSber(SimpleAssistant):
    def __init__(self, system_prompt):
        super().__init__(system_prompt)
    def generate_auth_data(self, user_id, secret):
        return {"user_id": user_id, "secret": secret}
    def initialize(self):
        return GigaChat(
            credentials=config.GIGA_CHAT_AUTH, 
            model="GigaChat-Pro",
            verify_ssl_certs=False,
            temperature=0.4,
            scope = config.GIGA_CHAT_SCOPE)
    
class SimpleAssistantGemini(SimpleAssistant):
    def __init__(self, system_prompt):
        super().__init__(system_prompt)

    def initialize(self):
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.4,
            google_api_key=config.GEMINI_API_KEY,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024
        )


