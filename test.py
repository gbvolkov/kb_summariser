# %%
import config
from assistants.json_assistants import JSONAssistantYA, JSONAssistantSber, JSONAssistantGPT, JSONAssistantMistralAI
from assistants.simple_assistants import SimpleAssistantSber

from pydantic import BaseModel, Field
from typing import List, Any, Optional, Dict, Tuple

import pandas as pd
import os

os.environ['CURL_CA_BUNDLE'] = '' 
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL'] = 'true'


# %%
class QA(BaseModel):
    question: Optional[str] = Field(description="The question in Russian.")
    answer: Optional[str] = Field(description="The answer in Russian.")

class QAs(BaseModel):
    """Convert a text document into a series of questions and answers."""

    qas: Optional[List[QA]] = Field(description="The list of questions and answers.")

# %%
#assistant = SimpleAssistantSber('Convert a text document into a series of questions and answers.')
assistant = JSONAssistantSber(QAs)

# %%
df = pd.read_csv('./data/articles_data.csv')
#df.head()
print(df.columns)

# %%
document = df['refs'][201]
print(document)

# %%
#os.environ["LANGCHAIN_TRACING_V2"] = "true"

result = assistant.ask_question(document)
print(result)