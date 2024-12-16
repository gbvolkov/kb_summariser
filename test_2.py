# %%
import config
from AIAssistantsLib.assistants import JSONAssistantYA, JSONAssistantSber, JSONAssistantGPT, JSONAssistantMistralAI
from AIAssistantsLib.assistants import SimpleAssistantSber, SimpleAssistantGPT

from pydantic import BaseModel, Field
from typing import List, Any, Optional, Dict, Tuple

import pandas as pd
import re
import os
import json

os.environ['CURL_CA_BUNDLE'] = '' 
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL'] = 'true'

os.environ["LANGCHAIN_TRACING_V2"] = "true"


with open('./data/terms&defs.txt', 'r', encoding='utf-8') as f:
    glossarry = f.read()

prompt = f"""Создайте резюме документа на русском языке. Используйте перечень Сокращения, Терминов и определений и Подразделений из контекста Glossary.

Glossary:
{glossarry}

"""

prompt_init = f"""Верните список используемых терминов и аббревиатур из документа пользователя.

"""

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


t_and_d = extract_terms_definitions(glossarry).copy()

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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

t_d_docs = dicts_to_documents(t_and_d)

embeddings = HuggingFaceEmbeddings(model_name='/models/multilingual-e5-large')
t_d_vs = FAISS.from_documents(t_d_docs, embeddings)


class Term(BaseModel):
    term: Optional[str] = Field(description="Термин или аббревиатура.")

class Terms(BaseModel):
    """Верните список используемых терминов и аббревиатур из документа пользователя."""

    terms: Optional[List[Term]] = Field(description="Список терминов или аббревиатур.")


class QA(BaseModel):
    question: Optional[str] = Field(description="The question in Russian.")
    answer: Optional[str] = Field(description="The answer in Russian.")

class QAs(BaseModel):
    """Convert a text document into a series of short questions and answers."""

    qas: Optional[List[QA]] = Field(description="The list of questions and answers.")


#assistant = SimpleAssistantSber('Convert a text document into a series of questions and answers.')
sber_assistant = SimpleAssistantSber(prompt)
structured = JSONAssistantGPT(Terms)
gpt_assistant = SimpleAssistantGPT(prompt)

def cleanup_text(text):
    return re.sub(r'##IMAGE##\s+\S+\.(png|jpg|jpeg|gif)', '', text)

def generate_summary(refs):
    json_string = structured.ask_question(cleanup_text(refs))
    list_of_terms = json.loads(json_string)
    docs = []
    for term in list_of_terms['terms']:
        found = t_d_vs.similarity_search_with_relevance_scores(term['term'], k = 20)
        found = sorted(found, key=lambda x: x.relevance, reverse=True)
        docs.extend(found)
    print(docs)
    try:
        summary = sber_assistant.ask_question(cleanup_text(refs))
    except Exception as e:
        summary = gpt_assistant.ask_question(cleanup_text(refs))
    return summary

df = pd.read_csv('./data/articles_data.csv')
#df.head()
print(df.columns)

df = df[26:26]

print(df)

# %%


df['json_summary'] = df['refs'].apply(generate_summary)

df.to_csv('./data/articles_data_summary.csv', index=False)

#document = df['refs'][201]
#print(document)
#result = assistant.ask_question(document)
#print(result)