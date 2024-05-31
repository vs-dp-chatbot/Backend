import pandas as pd
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key = OPENAI_API_KEY,
)

def tr_test(name, description,):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
            "content": "You are functioning as a keyword generator for translating the database table "},
            {"role": "system",
            "content": "You function to generate table name translation words by referring to the table description."},
            {"role": "system", "content": "You MUST create ONE keyword that translates the table name."},
            {"role": "system",
            "content": "Keyword should NOT be blank and it should be a Korean word."},
            {"role": "system", "content": "키워드는 영어를 제외하고 한국어로 생성한다. 키워드만 출력한다."},
            {"role": "user", "content": "table name:" + name + "table description:" + description + "\n"},
        ]
    )
    gpt = response.choices[0].message.content
    return gpt
def translate_text(name, description, source_lang="ko", target_lang="en"):

    # OpenAI 번역 API를 호출하는 함수
    gpt = tr_test(name, description)
    gptKeyword = check_keywords(name, description, gpt)
    return gptKeyword

def contains_english(keyword):
    pattern = re.compile(r'[a-zA-Z]')
    return bool(pattern.search(keyword))

def check_keywords(name, description, gptKeyword):
    keyword_length = len(gptKeyword)

    # 1. 길이 20 넘기면 재생성
    if keyword_length > 20:
        print(f"키워드 길이 초과: '{gptKeyword}'를 확인바랍니다.")
        return tr_test(name, description)

    # 2. 영어를 포함하고 있다면 재생성
    elif contains_english(gptKeyword):
        print(f"키워드 영어 포함: '{gptKeyword}'를 확인바랍니다.")
        return tr_test(name, description)

    else:
        return gptKeyword
    
def add_translation_row(csv_path, result_path):
    # CSV 파일 로드
    df = pd.read_csv(csv_path, encoding='utf-8')
    df.fillna('', inplace=True)
    df = df[['name','description']]
    df['name_ko'] = df.apply(lambda x: translate_text(delete(x['name']), x['description']), axis=1)
    df.to_csv(result_path, index=False)
    return df

def delete(name):
    return name.replace('_', ' ')

#----------------------------------------
def process_all_csv_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.csv'):
                print(filename)
                csv_path = os.path.join(root, filename)
                result_path = os.path.join(root, 'result_' + filename)
                add_translation_row(csv_path, result_path)
                print(f"Processing {csv_path} and saving result to {result_path}")


# 폴더 경로
folder_path = '컨텐츠Company'
process_all_csv_in_folder(folder_path)