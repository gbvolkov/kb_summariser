import config

from vs_utils import load_vectorstore

import re

import torch

from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, StringPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import FAISS
from langchain.schema.retriever import BaseRetriever 
from langchain.schema import HumanMessage, AIMessage, SystemMessage, Document
from langchain_community.llms import LlamaCpp
#from langchain_community.chat_models.gigachat import GigaChat
from langchain_gigachat import GigaChat
#from yandex_chain import YandexLLM
from langchain_community.llms import YandexGPT

#from langchain_community.llms import YandexGPT

#import tiktoken
from tiktoken import get_encoding, Encoding
#from ragatouille import RAGPretrainedModel
from langchain_community.document_compressors import JinaRerank
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import os

from abc import abstractmethod
from typing import List, Any, Optional, Dict, Tuple
import pickle

os.environ["LANGCHAIN_TRACING_V2"] = "true"

import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISTANCE_TRESHOLD = 0.8

#RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0") #Does not work on Windows. Use Jina instead.
#RERANKER = JinaRerank()
reranker_model = HuggingFaceCrossEncoder(model_name="models/bge-reranker-large")
RERANKER = CrossEncoderReranker(model=reranker_model, top_n=3)



class GPT2PromptTemplate(ChatPromptTemplate):
    def __init__(self):
        self.size=1024
    @property
    def size(self):
        return len(self.messages[0].content)


def show_retrieved_documents(vectorstore, retriever, query):
    import numpy as np
    results_with_scores = vectorstore.similarity_search_with_score(query, k=5)
    scores = [score for _, score in results_with_scores]

    logger.info(f"\n>>{query}==========================\n")
    # Check if documents have scores in metadata
    index = vectorstore.index  # FAISS index object

    # Check the metric
    metric = index.metric_type
    logger.info(f"FAISS Metric Type: {metric}\n")

    for doc, score in results_with_scores:
        doc_id = doc.metadata.get('problem_number')
        #score = 1 / (1 + score)
        description = doc.page_content
        idx_start = description.find('Problem Description')
        inx_end = description.find('Systems')
        logger.info(f"Document: {doc_id}: {description[idx_start:inx_end-1]}; ===>Similarity Score: {score}")
    logger.info(f"\n=========================={query}<<\n\n")



class ThresholdBasedRetriever(BaseRetriever):
    """
    A custom retriever that filters documents based on a distance threshold.
    """
    vectorstore: FAISS  # Type-annotated attribute
    distance_threshold: float
    k: int = 5  # Default value for top-k documents
    tokenizer: Encoding = get_encoding("o200k_base")
    max_tokens: int = -1

    def _count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text using the tokenizer.

        Parameters:
        - text (str): The text to tokenize.

        Returns:
        - int: Number of tokens.
        """
        return len(self.tokenizer.encode(text))
    
    def get_tokens_allocation(self, sorted_docs_and_scores: List[Tuple[Document, float]]) -> List[int]:
        # Calculate importance for each document (inverse of score)
        epsilon = 1e-6  # To prevent division by zero
        importances = [1 / (score + epsilon) for _, score in sorted_docs_and_scores]
        total_importance = sum(importances)

        # Number of documents
        num_docs = len(sorted_docs_and_scores)

        # Allocate tokens proportionally based on importance
        raw_allocations = [(imp / total_importance) * self.max_tokens for imp in importances]
        # Convert to integers
        allocated_tokens = [int(allocation) for allocation in raw_allocations]

        # Handle any remaining tokens due to integer rounding
        allocated_sum = sum(allocated_tokens)
        remaining_tokens = self.max_tokens - allocated_sum

        if remaining_tokens > 0:
            # Distribute the remaining tokens starting from the most important document
            for i in range(remaining_tokens):
                allocated_tokens[i % num_docs] += 1
        elif remaining_tokens < 0:
            # If over-allocated, remove tokens starting from the least important document
            for i in range(-remaining_tokens):
                idx = -(i % num_docs) - 1
                if allocated_tokens[idx] > 1:
                    allocated_tokens[idx] -= 1
        return allocated_tokens

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents relevant to the query based on the distance threshold
        and ensure the total tokens do not exceed max_tokens.

        Parameters:
        - query (str): The user query.

        Returns:
        - List[Document]: Filtered list of relevant documents within token limit.
        """
        # Perform similarity search with distance scores
        docs_and_scores: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_score(query, k=self.k)
        
        filtered_docs = []
        total_tokens = 0

        for doc, score in docs_and_scores:
            doc_id = doc.metadata.get('problem_number', 'N/A')
            
            # Since METRIC_L2 is used, lower scores are better
            if score <= self.distance_threshold:
                if self.max_tokens > 0:
                    doc_tokens = self._count_tokens(doc.page_content)
                    logger.info(f"Document ID: {doc_id}, Distance Score: {score}, Tokens: {doc_tokens}")
                    
                    if total_tokens + doc_tokens > self.max_tokens:
                        logger.info(f"Skipping Document ID: {doc_id} to maintain max_tokens limit.")
                        continue  # Skip this document as it would exceed the token limit
                    total_tokens += doc_tokens
                
                logger.info(f"Added Document ID: {doc_id}, Distance Score: {score} below threshold {self.distance_threshold}")
                filtered_docs.append(doc)
            else:
                logger.info(f"Rejected Document ID: {doc_id}, Distance Score: {score} above threshold {self.distance_threshold}")
        
        logger.info(f"Total tokens in returned documents: {total_tokens} (max allowed: {self.max_tokens})")
        return filtered_docs

"""
        for doc, score in docs_and_scores:
            doc_id = doc.metadata.get('problem_number', 'N/A')
            
            # Since METRIC_L2 is used, lower scores are better
            if score <= self.distance_threshold:
                doc_tokens = self._count_tokens(doc.page_content)
                if total_tokens + doc_tokens > self.max_tokens:
                    logger.info(f"Skipping Document ID: {doc_id} to maintain max_tokens limit.")
                    continue  # Skip this document as it would exceed the token limit

                logger.info(f"Accepted Document ID: {doc_id}, Distance Score: {score}")
                filtered_docs.append(doc)
                total_tokens += doc_tokens
            else:
                logger.info(f"Rejected Document ID: {doc_id}, Distance Score: {score} above threshold {self.distance_threshold}")
        return filtered_docs

                
"""        

class KBRetrieverManager:
    """
    Manager for handling multiple vector store retrievers.
    Ensures that each vector store is loaded only once and provides access to their retrievers.
    """

    def __init__(self, embedding_model: Optional[str] = None):
        """
        Initialize the KBRetrieverManager.  

        :param embedding_model: The embedding model to use. If None, uses config.EMBEDDING_MODEL.
        """
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.retrievers: Dict[(str, int), Any] = {}  # Maps vector_store_path and max_max_context_length to retriever
        self.vectorstores: Dict[str, Any] = {}  # Maps vector_store_path to vectorstore
        self.llm = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY, model_name="gpt-4o-mini") #os.getenv("OPENAI_API_KEY)
        logger.info(f"KBRetrieverManager initialized with embedding model: {self.embedding_model}")

    def get_retriever(self, vector_store_path: str, max_context_length: int = -1, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Retrieve the retriever for the specified vector store path.
        Loads the vector store if it hasn't been loaded yet.

        :param vector_store_path: Path to the vector store.
        :param search_kwargs: Keyword arguments for the retriever's search method.
        :return: The retriever object.
        """
        if (vector_store_path, max_context_length) in self.retrievers:
            logger.info(f"Retriever for '{vector_store_path}' already loaded. Returning existing retriever.")
            return self.retrievers[(vector_store_path, max_context_length)]

        logger.info(f"Loading vector store from '{vector_store_path}' with embedding model '{self.embedding_model}'.")
        try:
            if vector_store_path not in self.vectorstores:
                 (vectorstore, documents) = load_vectorstore(vector_store_path, self.embedding_model)
            else:
                (vectorstore, documents) = self.vectorstores[vector_store_path]
            #Load document store from persisted storage
            #loading list of problem numbers as ids
            doc_ids = [doc.metadata.get('problem_number', '') for doc in documents]
            store = InMemoryByteStore()
            id_key = "problem_number"
            multi_retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                byte_store=store,
                id_key=id_key,
                search_kwargs={"k": 5, "score_threshold": 0.8},
            )
            multi_retriever.docstore.mset(list(zip(doc_ids, documents)))
            retriever = ContextualCompressionRetriever(
                base_compressor=RERANKER, base_retriever=multi_retriever
                )

            self.retrievers[(vector_store_path, max_context_length)] = retriever
            self.vectorstores[vector_store_path] = (vectorstore, documents)
            logger.info(f"Vector store '{vector_store_path}' loaded and retriever initialized.")
            return retriever
        except Exception as e:
            logger.error(f"Failed to load vector store from '{vector_store_path}': {e}")
            raise e

    def reload_vector_store(self, vector_store_path: str, max_context_length: int = -1) -> Any:
        """
        Reload a specific vector store and update its retriever.

        :param vector_store_path: Path to the vector store.
        :return: The updated retriever object.
        """
        logger.info(f"Reloading vector store from '{vector_store_path}'.")
        try:
            (vectorstore, documents) = load_vectorstore(vector_store_path, self.embedding_model)
            doc_ids = [doc.metadata.get('problem_number', '') for doc in documents]
            store = InMemoryByteStore()
            id_key = "problem_number"
            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                byte_store=store,
                id_key=id_key,
                search_kwargs={"k": 5, "score_threshold": 0.8},
            )
            retriever.docstore.mset(list(zip(doc_ids, documents)))

            self.retrievers[(vector_store_path, max_context_length)] = retriever
            self.vectorstores[vector_store_path] = (vectorstore, documents)
            logger.info(f"Vector store '{vector_store_path}' reloaded and retriever updated.")
            return retriever
        except Exception as e:
            logger.error(f"Failed to reload vector store from '{vector_store_path}': {e}")
            raise e

    def unload_vector_store(self, vector_store_path: str, max_context_length: int = -1) -> None:
        """
        Unload a specific vector store and remove its retriever from the manager.

        :param vector_store_path: Path to the vector store.
        """
        if vector_store_path in self.retrievers:
            del self.vectorstores[vector_store_path]
            del self.retrievers[(vector_store_path, max_context_length)]
            logger.info(f"Vector store '{vector_store_path}' unloaded and retriever removed.")
        else:
            logger.warning(f"Attempted to unload non-loaded vector store '{vector_store_path}'.")

    def unload_all(self) -> None:
        """
        Unload all vector stores and clear all retrievers.
        """
        self.retrievers.clear()
        self.vectorstores.clear()
        logger.info("All vector stores have been unloaded and retrievers cleared.")

retriever_manager = KBRetrieverManager()

def get_retriever(kkb_path, max_context_window = -1):
    return retriever_manager.get_retriever(kkb_path, max_context_window)

class KBDocumentPromptTemplate(StringPromptTemplate):
    max_length : int = 0
    def __init__(self, max_length: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.max_length = max_length

    def format(self, **kwargs: Any) -> str:
        page_conetnt = kwargs.pop("page_content")
        problem_number = kwargs.pop("problem_number")
        chunk_size = kwargs.pop("actual_chunk_size")
        #here additional data could be retrieved based on problem_number
        result = page_conetnt
        if self.max_length > 0:
            result = result[:self.max_length]
        return result

    @property
    def _prompt_type(self) -> str:
        return "kb_document"

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class RAGAssistant:
    def __init__(self, system_prompt, kkb_path, max_context_window = -1, output_parser = BaseOutputParser):
        logger.info(f"Initializing model with: {kkb_path}")
        self.system_prompt = system_prompt
        self.max_context_window = max_context_window
        self.output_parser = output_parser
        self.retriever = get_retriever(kkb_path, self.max_context_window)
        logger.info(f"Dataretrieved built: {kkb_path}")
        self.llm = self.initialize()
        self.set_system_prompt(self.system_prompt)
        logger.info("Initialized")

    def truncate_context(self, context, question, max_tokens):
        # Default implementation: no truncation
        return context

    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self.prompt = self.get_prompt(self.system_prompt)

        def get_chat_prompt_template_length(chat_prompt: ChatPromptTemplate) -> int:
            total_length = 0
            for message in chat_prompt.messages:
                if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
                    total_length += len(message.prompt.template)
                else:
                    # For custom message types, fallback to string representation
                    total_length += len(str(message))
            return total_length

        max_length = self.max_context_window - get_chat_prompt_template_length(self.prompt)
        my_prompt = KBDocumentPromptTemplate(max_length, input_variables=["page_content", "problem_number", "actual_chunk_size"])

        docs_chain = create_stuff_documents_chain(self.llm, self.prompt, output_parser=self.output_parser(), document_prompt=my_prompt, document_separator='\n#EOD\n\n')
        self.rag_chain = create_retrieval_chain(self.retriever, docs_chain)

    def get_prompt(self, system_prompt):
        #return ChatPromptTemplate.from_template(system_prompt)
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
    @abstractmethod
    def initialize(self):
        """
            Initialize model here.
        """

    def ask_question(self, query: str) -> str:
        if self.rag_chain is None:
            logger.error("RAG chain not initialized")
            raise ValueError("Model or RAG chain not initialized.")
        try:
            #show_retrieved_documents(self.vectorstore, self.retriever, query)
            result = self.rag_chain.invoke({"input": query})
            return result #result['answer']
        except AttributeError as e:
            logger.error(f"AttributeError in ask_question: {str(e)}")
            raise ValueError(f"Error processing query: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in ask_question: {str(e)}")
            raise


class RAGAssistantGPT(RAGAssistant):
    def __init__(self, system_prompt, kkb_path, output_parser = BaseOutputParser):
        super().__init__(system_prompt, kkb_path, output_parser = output_parser)
    def initialize(self):
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4)

class RAGAssistantMistralAI(RAGAssistant):
    def __init__(self, system_prompt, kkb_path, output_parser = BaseOutputParser):
        super().__init__(system_prompt, kkb_path, output_parser = output_parser)
    def initialize(self):
        return ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.4)


class RAGAssistantYA(RAGAssistant):
    def __init__(self, system_prompt, kkb_path, output_parser = BaseOutputParser):
        super().__init__(system_prompt, kkb_path, 7200, output_parser = output_parser)
    def initialize(self):
        return YandexGPT(
            #iam_token = None,
            api_key = config.YA_API_KEY, 
            folder_id=config.YA_FOLDER_ID, 
            model_uri=f'gpt://{config.YA_FOLDER_ID}/yandexgpt/rc',
            temperature=0.4
            )

class RAGAssistantSber(RAGAssistant):
    def __init__(self, system_prompt, kkb_path, output_parser = BaseOutputParser):
        super().__init__(system_prompt, kkb_path, output_parser = output_parser)
    def generate_auth_data(self, user_id, secret):
        return {"user_id": user_id, "secret": secret}
    def initialize(self):
        return GigaChat(
            credentials=config.GIGA_CHAT_AUTH, 
            model="GigaChat-Pro",
            verify_ssl_certs=False,
            temperature=0.4,
            scope = config.GIGA_CHAT_SCOPE)
    
class RAGAssistantGemini(RAGAssistant):
    def __init__(self, system_prompt, kkb_path, output_parser = BaseOutputParser):
        super().__init__(system_prompt, kkb_path, output_parser = output_parser)

    def initialize(self):
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.4,
            google_api_key=config.GEMINI_API_KEY,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024
        )


def check_file_extension(filename, allowed_extensions):
    """
    Check if the file has an allowed extension.
    
    :param filename: The name or path of the file to check
    :param allowed_extensions: A list of allowed file extensions (e.g., ['.txt', '.pdf'])
    :return: True if the file has an allowed extension, False otherwise
    """
    # Get the file extension (convert to lowercase for case-insensitive comparison)
    _, file_extension = os.path.splitext(filename)
    file_extension = file_extension.lower()
    
    # Check if the file extension is in the list of allowed extensions
    return file_extension in [ext.lower() for ext in allowed_extensions]

class RAGAssistantLocal(RAGAssistant):
    def __init__(self, system_prompt, kkb_path, model_name='data/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf', output_parser = BaseOutputParser):
        self.model_name = model_name
        self.max_new_tokens = 2000
        #system_prompt = local_prompt #"Ты очень знающий сотрудник справочного бюро. Отвечай на вопросы легко и непринужденно. Question: {question} \nAnswer:"
        super().__init__(system_prompt, kkb_path, 4096, output_parser = output_parser)
        #self.rag_chain = (
        #    {"question": RunnablePassthrough()}
        #    | self.prompt
        #    | self.llm
        #    | StrOutputParser()
        #)
        

    def initialize(self):
        # Load the text generation model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")

        generation_config = GenerationConfig.from_pretrained(self.model_name)
        generation_config.max_new_tokens = 1024
        generation_config.temperature = 0.4
        generation_config.top_p = 0.9
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.2
        generation_config.eos_token_id=self.tokenizer.eos_token_id,
        generation_config.pad_token_id=self.tokenizer.eos_token_id

        pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer, generation_config=generation_config,)
        llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0.4})
        return llm
    
    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self.prompt = self.get_prompt(self.system_prompt)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        def truncate_chain(inputs):
            context = format_docs(inputs["context"])
            truncated_context = self.truncate_context(context, inputs["question"], self.max_context_window)
            return {
                "context": truncated_context,
                "question": inputs["question"]
            }
        
        def get_chat_prompt_template_length(chat_prompt: ChatPromptTemplate) -> int:
            total_length = 0
            for message in chat_prompt.messages:
                if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
                    total_length += len(message.prompt.template)
                else:
                    # For custom message types, fallback to string representation
                    total_length += len(str(message))
            return total_length

        max_length = self.max_context_window - get_chat_prompt_template_length(self.prompt)
        my_prompt = KBDocumentPromptTemplate(max_length, input_variables=["page_content", "problem_number", "actual_chunk_size"])
        docs_chain = create_stuff_documents_chain(self.llm, self.prompt, output_parser=self.output_parser(), document_prompt=my_prompt, document_separator='\n#EOD\n\n')
        self.rag_chain = create_retrieval_chain(self.retriever, docs_chain)

    def ask_question(self, query: str) -> str:
        if self.rag_chain is None:
            logger.error("RAG chain not initialized")
            raise ValueError("Model or RAG chain not initialized.")
        try:
            result = self.rag_chain.invoke({"input": query})
            with open("./debug/debug_local.txt", "w", encoding="utf-8") as f:
                f.write(result['answer'])
            return result
        except AttributeError as e:
            logger.error(f"AttributeError in ask_question: {str(e)}")
            raise ValueError(f"Error processing query: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in ask_question: {str(e)}")
            raise

    def count_tokens(self, text):
        if isinstance(self.llm, LlamaCpp):
            return self.llm.get_num_tokens(text)
        else:
            return len(self.tokenizer.encode(text))

    def truncate_input(self, text, max_tokens):
        if isinstance(self.llm, LlamaCpp):
            while self.count_tokens(text) > max_tokens:
                text = text[:int(len(text) * 0.9)]  # Reduce by 10% and retry
        else:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_tokens:
                text = self.tokenizer.decode(tokens[:max_tokens])
        return text

    def truncate_context(self, context, question, max_tokens):
        question_tokens = self.count_tokens(question)
        system_tokens = self.count_tokens(self.system_prompt)
        available_tokens = max_tokens - question_tokens - system_tokens - self.max_new_tokens
        return self.truncate_input(context, available_tokens)

class RAGAssistantGGUF(RAGAssistantLocal):
    def __init__(self, system_prompt, kkb_path, model_name='models/mistral-large-instruct-2411-Q4_K_M', output_parser = BaseOutputParser):
        super().__init__(system_prompt, kkb_path, model_name, output_parser)
        

    def initialize(self):
        # Load the text generation model and tokenizer
        logger.info(f"loading {self.model_name}...")
        try:
            llm = LlamaCpp(
                model_path=self.model_name,
                temperature=0.4,
                top_p=0.9,
                max_tokens=self.max_new_tokens,
                n_ctx=self.max_context_window,
                echo=False
            )
        except Exception as e:
            logger.error(f"Error loading {self.model_name}: {str(e)}")
        logger.info(f"...{self.model_name} loaded")
        self.tokenizer = None
        return llm



if __name__ == '__main__':
    from argparse import (
        ArgumentParser,
        ArgumentDefaultsHelpFormatter,
        BooleanOptionalAction,
    )
    from langchain_core.output_parsers import StrOutputParser

    vectorestore_path = 'data/vectorstore_e5'

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

    with open('prompts/system_prompt_markdown_3.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    if mode == 'query':
        assistants = []
        vectorstore = load_vectorstore(vectorestore_path, config.EMBEDDING_MODEL)
        retriever = get_retriever(vectorestore_path)
        assistants.append(RAGAssistantGPT(system_prompt, vectorestore_path, output_parser=StrOutputParser))

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
