# %%
import config
from AIAssistantsLib.assistants import JSONAssistantYA, JSONAssistantSber, JSONAssistantGPT, JSONAssistantMistralAI
from AIAssistantsLib.assistants import SimpleAssistantSber

from pydantic import BaseModel, Field
from typing import List, Any, Optional, Dict, Tuple

import pandas as pd
import os
import json

os.environ['CURL_CA_BUNDLE'] = '' 
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL'] = 'true'

os.environ["LANGCHAIN_TRACING_V2"] = "true"


class QA(BaseModel):
    question: Optional[str] = Field(description="The question in Russian.")
    answer: Optional[str] = Field(description="The answer in Russian.")

class QAs(BaseModel):
    """Convert a text document into a series of short questions and answers."""

    qas: Optional[List[QA]] = Field(description="The list of questions and answers.")


#assistant = SimpleAssistantSber('Convert a text document into a series of questions and answers.')
sber_assistant = JSONAssistantSber(QAs)
gpt_assistant = JSONAssistantGPT(QAs)


def generate_qas(refs):
    try:
        qas = gpt_assistant.ask_question(refs)
    except Exception as e:
        qas = sber_assistant.ask_question(refs)
    return qas

df = pd.read_csv('./data/articles_data.csv')
#df.head()
print(df.columns)

#df = df[26:29]

#print(df)

def parse_qas(json_string):
    data = json.loads(json_string)
    qa_list = []
    for qa in data['qas']:
        qa_list.append({
                'question': qa['question'],
                'answer': qa['answer']
            })
    return qa_list
# %%


df['json_qas'] = df['refs'].apply(generate_qas)
df['qas'] = df['json_qas'].apply(parse_qas)
df_exploded = df.explode('qas').reset_index(drop=True)
qa_df = pd.json_normalize(df_exploded['qas'])

df_final = pd.concat([df_exploded.drop(columns=['qas', 'json_qas']), qa_df], axis=1)

df_final.to_csv('./data/articles_data_qas.csv', index=False)

#document = df['refs'][201]
#print(document)
#result = assistant.ask_question(document)
#print(result)