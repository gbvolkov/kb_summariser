import config
import pandas as pd
import asyncio
import os 
import re
import sys
from typing import Any


from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

from utils.kb_summariser_v2 import get_summaries_generator

import logging


# Download the required NLTK data (you need this only once)
nltk.download('punkt_tab')

def cleanup_text(text):
    return re.sub(r'##IMAGE##\s+\S+\.(png|jpg|jpeg|gif)', '', text)

def validate_summaries_type(summaries):
    if not isinstance(summaries, list):
        return False
    return all(isinstance(item, dict) for item in summaries)
    

def summarise_text(text):
    text = cleanup_text(text)
    #return [{'summary': text[:256]}]
    summaries = generator(text)
    if not validate_summaries_type(summaries):
        summaries = [summaries]
    return summaries


generator = get_summaries_generator()
embedding_model_name = config.EMBEDDING_MODEL


def process_csv_chunked(input_path, output_path, chunk_size=4096, overlap=0.35, skiprows=None):
    with pd.read_csv(input_path, chunksize=1, encoding="utf-8", skiprows=skiprows) as reader:
        # Determine the chunk size and overlap size (35%)
        
        for chunk in reader:
            refs = chunk['refs'].iloc[0]  # Access the value in the 'refs' column
            overlap_size = int(chunk_size * overlap)

            # Generate overlapping chunks
            start = 0
            end = 0
            while end < len(refs):
                end = start + chunk_size
                text_chunk = refs[start:end]

                # Create a summary for the chunk
                summary = summarise_text(text_chunk)

                # Create a new row with the original chunk values and add the summary
                new_row = chunk.copy()
                new_row['solution'] = summary

                # Append the new row to the output CSV
                new_row.to_csv(output_path, mode='a', index=False, header=not pd.io.common.file_exists(output_path))

                # Move the start forward by chunk size minus the overlap
                start += chunk_size - overlap_size

def _len(text: str, tokenizer: Any = None) -> int:
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        return len(text)
    

def chunk_sentences(sentences, max_chunk_size, overlap=0, tokenizer=None):
    chunks = []
    current_chunk = []
    current_length = 0
    idx = 0
    overlap_size = 0

    while idx < len(sentences):
        full_sentence = sentences[idx]
        full_sentence_length = _len(full_sentence, tokenizer)
        if full_sentence_length > max_chunk_size:
            current_sentences = word_tokenize(full_sentence, language='russian')
        else:
            current_sentences = [full_sentence]
        
        for sentence in current_sentences:
            sentence_length = _len(sentence, tokenizer)

            if current_length + sentence_length <= max_chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Add the current chunk to the chunks list
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                # Determine the number of sentences to overlap based on overlap_size
                overlap_sentences = []
                overlap_length = 0
                overlap_idx = idx - 1
                
                while overlap_idx >= 0 and overlap_length + _len(sentences[overlap_idx], tokenizer) <= overlap_size:
                    overlap_sentences.insert(0, sentences[overlap_idx])
                    overlap_length += _len(sentences[overlap_idx], tokenizer)
                    overlap_idx -= 1
                
                # Start a new chunk with overlapping sentences
                current_chunk = overlap_sentences.copy()
                current_length = overlap_length

                overlap_size = int(sentence_length * overlap)

        idx += 1

    # Add any remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def cleanup_refs_for_processing(text: str):
    return text.replace('Загрузка, пожалуйста подождите.', '')
    #return re.sub(r'##IMAGE##\s+\S+\.(png|jpg|jpeg|gif)', '', text)



def process_csv(input_path, output_path, chunk_size=4096, overlap=0.35, skiprows=None, embedding_model_name: str = ''):

    tokenizer = None
    if embedding_model_name:
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    
    with pd.read_csv(input_path, chunksize=1, encoding="utf-8", skiprows=skiprows) as reader:
        # Determine the chunk size and overlap size (35%)
        idx = 0 if skiprows is None else skiprows.stop
        for chunk in reader:
            try:
                processed_df = pd.DataFrame()
                refs = str(chunk['refs'].iloc[0])  # Access the value in the 'refs' column
                refs = cleanup_refs_for_processing(refs)

                if _len(refs, tokenizer) >= chunk_size:
                    sentences = sent_tokenize(refs, language='russian')
                    #text_chunks = chunk_sentences(sentences, max_chunk_size=chunk_size, overlap_size=chunk_size * overlap, tokenizer=tokenizer)
                    text_chunks = chunk_sentences(sentences, max_chunk_size=chunk_size, overlap=overlap, tokenizer=tokenizer)
                else:
                    text_chunks = [refs]
                for text_chunk in text_chunks:
                    summaries = summarise_text(text_chunk)
                    for summary in summaries:
                        new_row = chunk.copy()
                        if 'summary' in summary:
                            solution = summary['summary']
                        else:
                            print(f"Error processing record {chunk['no'].iloc[0]}: No summary found.")
                            sys.exit(-1)
                        problem = f'{chunk['problem'].iloc[0]}: {summary['topic']}' if 'topic' in summary else chunk['problem'].iloc[0]
                        new_row['solution'] = solution
                        new_row['problem'] = problem
                        new_row['refs'] = text_chunk
                        #print(f'for Record NO: {chunk['no'].iloc[0]}: {problem}: {solution}')
                        processed_df = pd.concat([processed_df, new_row], ignore_index=True)
                processed_df.to_csv(output_path, mode='a', index=False, header=not pd.io.common.file_exists(output_path))
                #print(chunk['no'].iloc[0])
                with open('./data/summariser.idx', 'w', encoding='utf-8') as f:
                    f.write(f'{idx}')
                idx += 1
            except Exception as e:
                print(f"Error processing record {chunk['no'].iloc[0]}: {e}")
                continue

async def main():
    idx = 1
    if os.path.exists('./data/summariser.idx'):
        with open('./data/summariser.idx', 'r', encoding='utf-8') as f:
            idx = int(f.read())+1
    
    skiprows = None if idx == 1 else range(1, idx)
    
    process_csv('./data/articles_data.csv', './data/articles_data_summ.csv', chunk_size=8192, overlap=0.25, skiprows=skiprows, embedding_model_name=config.EMBEDDING_MODEL)

if __name__ == "__main__":
    asyncio.run(main())
