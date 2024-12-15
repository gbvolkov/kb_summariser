import os

# Add the parent directory to sys.path
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig

from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_gigachat import GigaChat
from langchain_community.llms import YandexGPT
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
#from langchain_core.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline



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
        #self.system_prompt = SystemMessage(content=prompt)
        self.system_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(prompt),
            ChatPromptTemplate.from_template('{query}')
            ])

    @abstractmethod
    def initialize(self):
        """
            Initialize model here.
        """

    def ask_question(self, query: Dict) -> str:
        if self.llm is None:
            logger.error("RAG chain not initialized")
            raise ValueError("Model or RAG chain not initialized.")
        try:
            chain = self.system_prompt | self.llm
            result = chain.invoke(query)
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
            #model="GigaChat",
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

class SimpleAssistantLocal(SimpleAssistant):
    def __init__(self, system_prompt, model_name='/models/llama3.1.8b'):
        self.model_name = model_name
        self.max_new_tokens = 2000
        super().__init__(system_prompt)
        

    def initialize(self):
        # Load the text generation model and tokenizer
        if torch.cuda.is_available():
            torch_dtype = torch.float16
            device = "cuda"  
        else:
            torch_dtype = torch.float32
            device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, trust_remote_code=True, device_map="auto")

        generation_config = GenerationConfig.from_pretrained(self.model_name)
        generation_config.max_new_tokens = 1024
        generation_config.temperature = 0.4
        generation_config.top_p = 0.9
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.2
        generation_config.eos_token_id=self.tokenizer.eos_token_id,
        generation_config.pad_token_id=self.tokenizer.eos_token_id

        pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer, generation_config=generation_config)
        logging.info(f'Using device: {pipe.device}')
        llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0.4})
        return llm


if __name__ == '__main__':
    from argparse import (
        ArgumentParser,
        ArgumentDefaultsHelpFormatter,
        BooleanOptionalAction,
    )
    from langchain_core.output_parsers import StrOutputParser

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'mode', 
        nargs='?', 
        default='query', 
        choices = ['query'],
        help='query - query vectorestore\n'
    )

    args = vars(parser.parse_args())
    mode = args['mode']
    model = '/models/llama3.1.8b'
    system_prompt = "Ты внимательный собеседник"

    if mode == 'query':
        assistants = []
        assistants.append(SimpleAssistantLocal(system_prompt))

        query = ''

        while query != 'stop':
            print('=========================================================================')
            query = input("Enter your query: ")
            if query != 'stop':
                for assistant in assistants:
                    try:
                        reply = assistant.ask_question(query)
                    except Exception as e:
                        logging.error(f'Error: {str(e)}')
                        continue
                    print(f'{reply['answer']}')
                    print('=========================================================================')


    
