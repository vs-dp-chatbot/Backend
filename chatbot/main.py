from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

class Item(BaseModel):
    company: str
    domain: Optional[str] = None
    message: str

@app.post("/home")
async def answer(item: Item):
    dic = {"컨텐츠":"vs-dp-content", "러닝":"vs-dp-learning"}
    import os
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SERVER_URL = os.getenv("SERVER_URL")

    import chromadb
    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings
    client = chromadb.HttpClient(host=SERVER_URL, port=8001)
    embedding = OpenAIEmbeddings()
    db = Chroma(client=client, collection_name=dic[item.company], embedding_function=embedding)
    
    if item.domain:
        db.get(where={"source": {"$in": [item.domain]}})
    
    from langchain.chat_models import ChatOpenAI
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=OPENAI_API_KEY)

    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    
    system_template="""
    You are functioning to recommend the database table.
    교육 도메인에서 참고하면 좋을 데이터 테이블에는 어떤게 있을지 키워드 4가지 알려줘.
    키워드는 ,로 분리해서 오직 키워드들 4가지만 출력해줘.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    from langchain.chains import LLMChain
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(question=item.message)
    response = response.replace("\n", "")

    answer = {}
    elements = response.split(", ")  # 문자열을 ", "를 기준으로 분리하여 리스트로 저장
    print(elements)
    for element in elements:
        print(element)
        retrieved_pages = db.similarity_search(element, k=4)
        for p in retrieved_pages:
            key = p.page_content.split('\n')[0].split(': ')[1]
            print(key)
            if key not in answer:
                answer[key] = {
                    "description": p.page_content.split('\n')[1].split(': ')[1],
                    "name_ko": p.page_content.split('\n')[2].split(': ')[1],
                    "company": p.metadata['source'].split('\\')[0],
                    "domain": p.metadata['source'].split('\\')[1], 
                    "urn": os.getenv("URL") + p.metadata['urn'],
                }
    return answer
