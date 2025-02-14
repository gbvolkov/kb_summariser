import config

from AIAssistantsLib.assistants import JSONAssistantYA, JSONAssistantSber, JSONAssistantGPT, JSONAssistantMistralAI, SimpleAssistantSber, SimpleAssistantGPT, SimpleAssistantMistralAI

from pydantic import BaseModel, Field
from typing import List, Any, Optional, Dict, Tuple

import time
import pandas as pd
import re
import os
import json
import sys

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser

os.environ["LANGCHAIN_TRACING_V2"] = "false"



# Определение структур возвращаемых данных
class Summary(BaseModel):
    """Наименование документа (topic) и краткое описание (summary) документа на русском языке."""

    topic: str = Field(description="Topic which summary is applied to")
    summary: str = Field(description="Short summary of the article for the topic")

class Summaries(BaseModel):
    """Список возможных наименований темы документа (topic) и кратких описаний (summary) темы документа на русском языке."""

    summaries: List[Summary] = Field(description="List of summaries of a specific document theme")

class Term(BaseModel):
    term: Optional[str] = Field(description="Термин или аббревиатура.")

class Terms(BaseModel):
    """Верните список используемых терминов и аббревиатур из документа пользователя."""

    terms: Optional[List[Term]] = Field(description="Список терминов или аббревиатур.")


def extract_terms_definitions(text: str) -> List[Dict[str, str]]:
    """
    Extracts terms and their definitions from a structured text document.

    Args:
        text (str): The input text containing terms and definitions in the following format:
                    ##Term: <Term>
                    ##Definition: <Definition>

    Returns:
        List[Dict[str, str]]: A list of dictionaries with 'term' and 'definition' keys.
    """
    # Define the regular expression pattern
    pattern = r'##Term:\s*(.*?)\s*##Definition:\s*(.*?)(?=##Term:|$)'

    # Use re.findall to extract all matches
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    # Process matches to create a list of dictionaries
    terms_definitions = []
    for term, definition in matches:
        # Clean up the extracted term and definition by stripping whitespace
        term_clean = term.strip()
        definition_clean = definition.strip()

        # Append to the list as a dictionary
        terms_definitions.append({
            'term': term_clean,
            'definition': definition_clean
        })

    return terms_definitions

def dicts_to_documents(terms_definitions: List[Dict[str, str]]) -> List[Document]:
    """
    Transforms a list of dictionaries with 'term' and 'definition' into a list of LangChain Documents.

    Args:
        terms_definitions (List[Dict[str, str]]): List of dictionaries containing 'term' and 'definition'.

    Returns:
        List[Document]: List of LangChain Document objects.
    """
    documents = []
    for entry in terms_definitions:
        term = entry.get('term', '').strip()
        definition = entry.get('definition', '').strip()
        
        # Create a Document with definition as content and term as metadata
        doc = Document(
            page_content=f'{term}: {definition}',
            metadata={"term": term}
        )
        documents.append(doc)
    
    return documents


def cleanup_text(text: str):
    #text = text.replace('Загрузка, пожалуйста подождите.', '')
    return re.sub(r'##IMAGE##\s+\S+\.(png|jpg|jpeg|gif)', '', text)


def get_summaries_generator():
    #Читаем список терминов и определений. TODO: задавать имя файла параметром
    with open('./data/terms&defs.txt', 'r', encoding='utf-8') as f:
        glossary = f.read()


    #Системный промпт для подготовки summary. TODO: получать из config или задавать параметром
    prompt = """Прочитайте и проанализируйте инструкцию, разработанную для специалистов техподдержки для ответов на вопросы. 
    На основе анализа выделите системы и основные темы, которые затрагивает инструкция. 
    Для каждой темы создайте наименование документа (topic) и краткое описание (summary) документа на русском языке, по которыму можно будет легко найти текст по запросу пользователя. 
    Используйте перечень Сокращения, Терминов и определений и Подразделений из контекста Glossary.

    Возвращайте результат в виде массива JSON с полями topic и summary.

    Glossary:
    {glossary}

    """

    #loading embedding model
    embeddings = HuggingFaceEmbeddings(model_name='/models/multilingual-e5-large')
    #Creating vectore store for terms and definitions
    t_d_vs = FAISS.from_documents(dicts_to_documents(extract_terms_definitions(glossary)), embeddings)

    sber_assistant = SimpleAssistantMistralAI(prompt)
    structured = JSONAssistantMistralAI(Terms)
    structured_gpt = JSONAssistantGPT(Terms)
    gpt_assistant = SimpleAssistantGPT(prompt)

    parser = JsonOutputParser(pydantic_object=Summary)

    def generate_summary(refs):
        cleaned_refs = cleanup_text(refs)
        
        # Extracts list of terms from the document
        try:
            json_string = structured.ask_question(cleaned_refs)
        except Exception as e:
            json_string = structured_gpt.ask_question(cleaned_refs)
        list_of_terms = json.loads(json_string)
        list_of_terms['terms'] = [item for item in list_of_terms['terms'] if item['term'] is not None]
        docs = []
        # Retrieves terms and definitions from t&d vectore store
        for term in list_of_terms['terms']:
            found = t_d_vs.similarity_search_with_relevance_scores(term['term'], k = 20)
            found = [doc.page_content for doc, score in found if score >= 0.78]
            docs.extend(found)
        glossary_context = "\n\n".join(doc for doc in docs)
        
        query = {
            "glossary": glossary_context,
            "query": cleaned_refs
        }
        # Generates list of topics and summaries
        ctry  = 3
        delay  = 1
        while ctry > 0:
            try:
                summary_txt = sber_assistant.ask_question(query)
                summary = parser.parse(summary_txt)
                return summary
            except Exception as e:
                time.sleep(delay)
                delay = delay * 3
                ctry = ctry - 1
        try:
            summary_txt = gpt_assistant.ask_question(query)
            summary = parser.parse(summary_txt)
        except:
            summary = query[:256]
        return summary
    
    return generate_summary