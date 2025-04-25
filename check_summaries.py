# %%
import pandas as pd
import re
import os
import config

from AIAssistantsLib.assistants import SimpleAssistantMistralAI, SimpleAssistantGPT, SimpleAssistantYA, SimpleAssistantSber

os.environ["LANGCHAIN_TRACING_V2"] = "false"

def cleanup_text(text: str):
    text = text.replace('Загрузка, пожалуйста подождите.', '')
    return re.sub(r'##IMAGE##\s+\S+\.(png|jpg|jpeg|gif)', '', text)


prompt = """Проанализируй текст. Верни OK, если текст содержит полезную информацию. Верни NOK, если текст не содержит полезной информации. Всегда возвращай только одно слово OK или NOK.
    """
checker = SimpleAssistantGPT(prompt)

write_header = True
with pd.read_csv('./data/articles_data_summ_cleaned.csv', chunksize=1, encoding="utf-8") as reader:
    idx = 0
    prev_refs = ''
    for chunk in reader:
        row_index = chunk.index[0]
        refs = chunk['refs'].iloc[0]
        if refs and type(refs) == str:
            no = chunk['no'].iloc[0]
            problem = chunk['problem'].iloc[0]
            cleaned_refs = cleanup_text(refs)
            query = {
                "query": cleaned_refs
            }
            if prev_refs != refs:
                validation = checker.ask_question(query=query)
                prev_refs = refs
            validated_row = chunk.copy()
            validated_row['validated'] = validation
            validated_row.to_csv(
                    './data/articles_data_summ_preprod.csv',
                    mode='w' if write_header else 'a',
                    header=write_header,
                    index=False
                )
            write_header = False
        else:
            print(f'Not updated{row_index}')
        idx += 1



