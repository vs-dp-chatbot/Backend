#https://github.com/zylon-ai/private-gpt/discussions/902 충돌문제 해결방법
#aws서버 연걸 https://docs.trychroma.com/deployment/aws
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from chromadb.utils import embedding_functions
from chromadb import HttpClient

client = HttpClient(host="43.203.123.103", port=8000)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-ada-002"
        )
print(client.list_collections())
collection = client.create_collection(name="vs-dp-content")
collection = client.get_collection(name="vs-dp-content", embedding_function=openai_ef)
print("collection.count: ", collection.count())

import uuid
import pandas as pd

def insertDB(csv_path):
    from langchain_community.document_loaders import CSVLoader
    loader = CSVLoader(file_path=csv_path, encoding='utf-8')
    data = loader.load()
    from langchain.text_splitter import RecursiveCharacterTextSplitter #단어 단위로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(data)

    df = pd.read_csv(csv_path.replace('result_', ''), encoding='utf-8')
    urn = df['urn']
    
    #적재
    for i, (doc, meta) in enumerate(zip(docs, urn)):
        try:
            metadata = {'urn': meta}
            metadata.update(doc.metadata)
            
            collection.add(
                ids=[str(uuid.uuid1())],
                metadatas=[metadata],  
                documents=[doc.page_content],
            )
        except Exception as e:
            print(f"Error adding document {i}: {e}")

#---------------------------------------
def process_all_csv_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if 'result_' in filename and filename.endswith('.csv'):
                print(filename)
                csv_path = os.path.join(root, filename)
                insertDB(csv_path)
                print(f"Processing {csv_path}")
                print("collection.count()", collection.count())
            
# 폴더 경로
folder_path = '컨텐츠Company'
process_all_csv_in_folder(folder_path)

