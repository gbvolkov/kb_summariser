import os
import time

# Add the parent directory to sys.path
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

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


class JSONAssistant:
    def __init__(self, schema):
        logger.info("Initializing model")
        self.schema = schema
        self.llm = self.initialize()
        self.set_system_prompt()
        logger.info("Initialized")

    def set_system_prompt(self):
        logger.info("set structured chain")
        self.chain = self.llm.with_structured_output(self.schema)

    @abstractmethod
    def initialize(self):
        """
            Initialize model here.
        """

    def ask_question(self, query: str) -> str:
        err_str = ''
        if self.chain is None:
            logger.error("RAG chain not initialized")
            raise ValueError("Model or RAG chain not initialized.")
        try:
            #show_retrieved_documents(self.vectorstore, self.retriever, query)
            cattempts = 0
            while cattempts < 5:
                cattempts += 1
                try:
                    result = self.chain.invoke(query)
                    return result.json()
                except Exception as e:
                    logger.error(f"Error in ask_question: {str(e)}")
                    err_str = f"Error in ask_question: {str(e)}"
                    time.sleep(1.1)
            raise ValueError(err_str)
        except AttributeError as e:
            logger.error(f"AttributeError in ask_question: {str(e)}")
            raise ValueError(f"Error processing query: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in ask_question: {str(e)}")
            raise


class JSONAssistantGPT(JSONAssistant):
    def __init__(self, schema):
        super().__init__(schema)
    def initialize(self):
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1)


class JSONAssistantMistralAI(JSONAssistant):
    def __init__(self, schema):
        super().__init__(schema)
    def initialize(self):
        return ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.1)


class JSONAssistantYA(JSONAssistant):
    def __init__(self, schema):
        super().__init__(schema)
    def initialize(self):
        return YandexGPT(
            api_key = config.YA_API_KEY, 
            folder_id=config.YA_FOLDER_ID, 
            model_uri=f'gpt://{config.YA_FOLDER_ID}/yandexgpt/latest',
            temperature=0.1
            )

class JSONAssistantSber(JSONAssistant):
    def __init__(self, schema):
        super().__init__(schema)
    def generate_auth_data(self, user_id, secret):
        return {"user_id": user_id, "secret": secret}
    def initialize(self):
        return GigaChat(
            credentials=config.GIGA_CHAT_AUTH, 
            model="GigaChat-Pro",
            #model="GigaChat",
            verify_ssl_certs=False,
            temperature=0.4,
            scope = config.GIGA_CHAT_SCOPE)
    
class JSONAssistantGemini(JSONAssistant):
    def __init__(self, system_prompt, kkb_path):
        super().__init__(system_prompt, kkb_path)

    def initialize(self):
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.1,
            google_api_key=config.GEMINI_API_KEY,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024
        )



if __name__ == '__main__':
    from typing import Optional
    from pydantic import BaseModel, Field

    class Procurement(BaseModel):
        """Procurement data"""
        supplier: Optional[str] = Field(default=None)
        good: Optional[str] = Field(default=None)
        good_volume: Optional[str] = Field(default=None)
        good_price: Optional[str] = Field(default=None)
        supply_cost: Optional[str] = Field(default=None)

    class Shipment(BaseModel):
        """Shipment data"""
        shipment_date: Optional[str] = Field(default=None)
        shipment_time: Optional[str] = Field(default=None)
        customer_name: Optional[str] = Field(default=None)
        customer_address: Optional[str] = Field(default=None)
        good: Optional[str] = Field(default=None)
        good_volume: Optional[str] = Field(default=None)
        good_price: Optional[str] = Field(default=None)
        shipment_count: Optional[str] = Field(default=None)
        shipment_cost: Optional[str] = Field(default=None)
        supplier: Optional[str] = Field(default=None)
        procurements: Optional[List[Procurement]] = Field(default=None)

    class Shipments(BaseModel):
        """List of shipments"""
        shipments: List[Shipment] = Field(description="List of shipments")


    assistant = JSONAssistantSber(schema=Shipments)
    text = " вот смотрите Георгий получается ну вот как бы мне нужно дать задание боту чтобы он бот запомни отгрузку создай новую отгрузку на 14.00 7 ноября по адресу ярославская шоссе дом 114 везем организации мастер строй тощи бетон марки 220 кубов по 4850 рублей одна доставка по 7 тысяч от организации евробетон потом дополнительно вторая ну вторая например и добавь туда закупку везем от организации авира строй везем марку 220 кубов по такой-то цени и так далее"
    result = assistant.ask_question(text)
    print(result)