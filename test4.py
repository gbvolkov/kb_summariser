# %%
import config
from assistants.json_assistants import JSONAssistantYA, JSONAssistantSber, JSONAssistantGPT, JSONAssistantMistralAI
from assistants.simple_assistants import SimpleAssistantSber, SimpleAssistantGPT

from pydantic import BaseModel, Field
from typing import List, Any, Optional, Dict, Tuple

import pandas as pd
import re
import os
import json
import sys

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser

os.environ["LANGCHAIN_TRACING_V2"] = "true"

# %%
with open('./data/terms&defs.txt', 'r', encoding='utf-8') as f:
    glossary = f.read()

prompt = """Создайте наименование документа (topic) и краткое описание (summary) документа на русском языке. 
Используйте перечень Сокращения, Терминов и определений и Подразделений из контекста Glossary.

Возвращайте результат в виде JSON с полями topic и summary.

Glossary:
{glossary}

"""

prompt_init = """Верните список используемых терминов и аббревиатур из документа пользователя.

"""

class Summary(BaseModel):
    """Наименование документа (topic) и краткое описание (summary) документа на русском языке."""
    topic: str = Field(description="Topic which summary is applied to")
    summary: str = Field(description="Short summary of the article for the topic")

class QA(BaseModel):
    question: Optional[str] = Field(description="The question in Russian.")
    answer: Optional[str] = Field(description="The answer in Russian.")

class QAs(BaseModel):
    """Convert a text document into a series of short questions and answers."""

    qas: Optional[List[QA]] = Field(description="The list of questions and answers.")


class Term(BaseModel):
    term: Optional[str] = Field(description="Термин или аббревиатура.")

class Terms(BaseModel):
    """Верните список используемых терминов и аббревиатур из документа пользователя."""

    terms: Optional[List[Term]] = Field(description="Список терминов или аббревиатур.")

# %%
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

# %%
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


# %%
t_and_d = extract_terms_definitions(glossary).copy()

# %%
t_d_docs = dicts_to_documents(t_and_d)
embeddings = HuggingFaceEmbeddings(model_name='/models/multilingual-e5-large')
t_d_vs = FAISS.from_documents(t_d_docs, embeddings)


# %%
sber_assistant = SimpleAssistantSber(prompt)
structured = JSONAssistantMistralAI(Terms)
gpt_assistant = SimpleAssistantGPT(prompt)

# %%
def cleanup_text(text: str):
    text = text.replace('Загрузка, пожалуйста подождите.', '')
    return re.sub(r'##IMAGE##\s+\S+\.(png|jpg|jpeg|gif)', '', text)

# %%
parser = JsonOutputParser(pydantic_object=Summary)

def generate_summary(refs):
    
    json_string = structured.ask_question(cleanup_text(refs))
    list_of_terms = json.loads(json_string)
    docs = []
    for term in list_of_terms['terms']:
        found = t_d_vs.similarity_search_with_relevance_scores(term['term'], k = 20)
        found = [doc.page_content for doc, score in found if score >= 0.78]
        #print(f'\n{term["term"]}:\n{found}\n')
        #found = sorted(found, key=lambda x: x.relevance, reverse=True)
        docs.extend(found)
    glossary_context = "\n\n".join(doc for doc in docs)
    #print(glossary_context)
    
    query = {
        "glossary": glossary_context,
        "query": cleanup_text(refs)
    }
    try:
        summary_txt = sber_assistant.ask_question(query)
        summary = parser.parse(summary_txt)
    except Exception as e:
        summary_txt = gpt_assistant.ask_question(query)
        summary = parser.parse(summary_txt)
    return summary


# %%
# Define file paths
input_csv_path = './data/articles_data_summ.csv'
output_csv_path = './data/articles_data_summarised.csv'
batch_size = 1  # Number of records to process in each batch

# %%
# Function to determine the starting point
def get_starting_index(output_path: str) -> int:
    """
    Determines the number of records already processed by checking the output CSV.

    Args:
        output_path (str): Path to the output CSV file.

    Returns:
        int: Number of records already processed.
    """
    if not os.path.exists(output_path):
        return 0
    existing_df = pd.read_csv(output_path)
    return len(existing_df)

# %%
def main():
    """
    Main function to process the CSV in batches and save progress incrementally.
    """
    starting_index = get_starting_index(output_csv_path)
    print(f"Starting processing from index: {starting_index}")

    # Read the input CSV in chunks
    chunk_iterator = pd.read_csv(input_csv_path, chunksize=batch_size, skiprows=range(1, starting_index + 1))

    # Initialize a flag to check if the output CSV needs headers
    write_header = not os.path.exists(output_csv_path)

    for chunk_number, chunk in enumerate(chunk_iterator, start=starting_index // batch_size + 1):
        print(f"\nProcessing batch {chunk_number} (rows {starting_index} to {starting_index + len(chunk)})")

        # Reset the index of the chunk to align with the original DataFrame
        chunk.reset_index(drop=True, inplace=True)

        # Apply the generate_summary function to each 'refs' entry
        try:
            summaries = chunk['refs'].apply(lambda x: pd.Series(generate_summary(x)))
            chunk[['title', 'summary']] = summaries
        except Exception as e:
            print(f"Error processing batch {chunk_number}: {e}")
            print("Stopping the processing. You can resume by rerunning the script.")
            sys.exit(1)

        # Select only the necessary columns (original plus new summaries)
        # Adjust the columns as per your requirements
        output_columns = list(chunk.columns) + ['title', 'summary']

        # Append the processed chunk to the output CSV
        if write_header:
            # Write with header if the file is being created
            chunk.to_csv(output_csv_path, mode='w', index=False)
            write_header = False
            print(f"Batch {chunk_number} saved to {output_csv_path} with headers.")
        else:
            # Append without header
            chunk.to_csv(output_csv_path, mode='a', index=False, header=False)
            print(f"Batch {chunk_number} appended to {output_csv_path}.")

        # Update the starting index
        starting_index += len(chunk)

    print("\nProcessing completed successfully.")

# %%
if __name__ == "__main__":
    main()