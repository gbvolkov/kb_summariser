
import config
from assistants.json_assistants import JSONAssistantYA, JSONAssistantSber, JSONAssistantGPT, JSONAssistantMistralAI
from assistants.simple_assistants import SimpleAssistantSber, SimpleAssistantGPT

from pydantic import BaseModel, Field
from typing import List, Any, Optional, Dict, Tuple

import pandas as pd
import re
import os
import json

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
os.environ["LANGCHAIN_TRACING_V2"] = "true"


with open('./data/terms&defs.txt', 'r', encoding='utf-8') as f:
    glossary = f.read()

prompt = """Создайте резюме документа на русском языке. Используйте перечень Сокращения, Терминов и определений и Подразделений из контекста Glossary.

Glossary:
{glossary}

"""

prompt_init = """Верните список используемых терминов и аббревиатур из документа пользователя.

"""

sber_assistant = SimpleAssistantSber(prompt)
gpt_assistant = SimpleAssistantGPT(prompt)

def cleanup_text(text):
    return re.sub(r'##IMAGE##\s+\S+\.(png|jpg|jpeg|gif)', '', text)

def generate_summary(refs):
    list_of_terms = ['ДКП']
    docs = ['БД: база данных', 'ДКП: договор-купли продажи']
    glossary_context = "\n\n".join(doc for doc in docs)
    print(glossary_context)
    query = {
        "glossary": glossary_context,
        "query": cleanup_text(refs)
    }
    try:
        summary = sber_assistant.ask_question(query)
    except Exception as e:
        summary = gpt_assistant.ask_question(query)
    return summary

df = pd.read_csv('./data/articles_data.csv')
df = df[26:29]

df['json_summary'] = df['refs'].apply(generate_summary)