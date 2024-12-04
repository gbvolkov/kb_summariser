import pandas as pd
import asyncio

from utils.kb_summariser import summarise, summarise_ya, summarise_chunked
import logging
import re
from nltk.tokenize import sent_tokenize
import nltk


# Download the required NLTK data (you need this only once)
nltk.download('punkt_tab')

def cleanup_text(text):
    return re.sub(r'##IMAGE##\s+\S+\.(png|jpg|jpeg|gif)', '', text)

def summarise_text(text):
    text = cleanup_text(text)
    return summarise_ya(text, max_length=1024, min_length=64, do_sample=False)


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


def chunk_sentences(sentences, max_chunk_size, overlap_size=0):
    chunks = []
    current_chunk = []
    current_length = 0
    idx = 0

    while idx < len(sentences):
        sentence = sentences[idx]
        sentence_length = len(sentence)
        
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
            
            while overlap_idx >= 0 and overlap_length + len(sentences[overlap_idx]) <= overlap_size:
                overlap_sentences.insert(0, sentences[overlap_idx])
                overlap_length += len(sentences[overlap_idx])
                overlap_idx -= 1
            
            # Start a new chunk with overlapping sentences
            current_chunk = overlap_sentences.copy()
            current_length = overlap_length
            
            # Avoid infinite loop by not resetting idx beyond a reasonable point
            #if not overlap_sentences:
                # If no overlap is possible, skip the problematic sentence
        idx += 1

    # Add any remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_csv(input_path, output_path, chunk_size=4096, overlap=0.35, skiprows=None):
    with pd.read_csv(input_path, chunksize=1, encoding="utf-8", skiprows=skiprows) as reader:
        # Determine the chunk size and overlap size (35%)

        for chunk in reader:
            refs = chunk['refs'].iloc[0]  # Access the value in the 'refs' column

            if len(refs) >= chunk_size:
                sentences = sent_tokenize(refs, language='russian')
                text_chunks = chunk_sentences(sentences, max_chunk_size=chunk_size, overlap_size=chunk_size * overlap)
            else:
                text_chunks = [refs]
            for text_chunk in text_chunks:
                summaries = summarise_text(text_chunk)
                for summary in summaries:
                    new_row = chunk.copy()
                    if 'summary' in summary:
                        solution = summary['summary']
                    else:
                        solution = summarise_chunked(text_chunk, max_length=chunk_size, min_length=256, do_sample=False)
                    problem = summary['topic'] if 'topic' in summary else chunk['problem'].iloc[0]
                    new_row['solution'] = solution
                    new_row['problem'] = problem
                    new_row['refs'] = text_chunk
                    print(f'for Record NO: {chunk['no'].iloc[0]}: {problem}: {solution}')
                    new_row.to_csv(output_path, mode='a', index=False, header=not pd.io.common.file_exists(output_path))

"""
            overlap_size = int(chunk_size * overlap)

            # Generate overlapping chunks
            start = 0
            end = 0
            while end < len(refs):
                end = start + chunk_size
                text_chunk = refs[start:end]

                # Create a summary for the chunk
                summaries = summarise_ya(text_chunk)
                for summary in summaries:
                    new_row = chunk.copy()
                    new_row['solution'] = summary['summary']
                    new_row['problem'] = summary['topic']
                    print(f'for Record NO: {chunk['no'].iloc[0]}: {summary["topic"]}: {summary["summary"]}')
                    new_row.to_csv(output_path, mode='a', index=False, header=not pd.io.common.file_exists(output_path))

                # Move the start forward by chunk size minus the overlap
                start += chunk_size - overlap_size
"""



async def main():
    process_csv('./output/articles_data.csv', './output/articles_data_summ.csv', chunk_size=8192, overlap=0.25)#, skiprows=range(1,140))

if __name__ == "__main__":
    asyncio.run(main())
