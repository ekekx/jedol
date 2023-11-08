

import json
import datetime
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader,WebBaseLoader,UnstructuredURLLoader
from langchain.text_splitter import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from dotenv import load_dotenv;load_dotenv() # openai_key  .env 선언 사용 
import jedol1Fun as jshs



# AI 역할
documents=[ Document(
            page_content=f"""
            오늘 일자는 {jshs.today_date()}이다.
            오늘 요일는 {jshs.today_week_name()}이다.
            이번 달은 {jshs.today_month()}이다.
            나는 제주과학고등학교 1학년에 재학 중입니다.
            나의 이름은 '제돌이'입니다.
            나의 나이는 16세입니다.
            나는 인공지능 기초를 재미있게 배우고 있습니다. 
            나는 코딩을 잘 합니다.
            학교에 관련된 궁금한 점이 있으면 질문하십시요.\n\n""", 
            metadata={'source': 'jedolStory'}
            )
]


# jshs.loader_documents_viewer(documents); quit()

# 학교연혁 
# txt 파일 사용용
# loader = TextLoader("data/history.txt", encoding='utf-8')
# html 사용
loader = WebBaseLoader(web_path="https://jeju-s.jje.hs.kr/jeju-s/0102/history")

page=loader.load()[0]

page.page_content=jshs.html_parsing_text(
                page_content=page.page_content,
                start_str="학교연혁 연혁 기본 리스트 년 도 날 짜 내 용",
                end_str="열람하신 정보에 대해 만족하십니까",
                length=20,
                removeword=[]
                )

documents.append( Document(
                    page_content=page.page_content,
                    metadata=page.metadata
                    )
                )

# print(page)
# quit()
# 주소
loader = WebBaseLoader(web_path="https://jeju-s.jje.hs.kr/jeju-s/0102/history")
page=loader.load()[0]
page.page_content=jshs.html_parsing_text(
                page_content=page.page_content,
                start_str="우[",
                end_str="Copyright",
                length=20,
                removeword=[]
                )
documents.append( Document(
                    page_content=page.page_content,
                    metadata=page.metadata
                    )
                )

# 식단-----------------------
from datetime import datetime, timedelta

today = datetime.now().today()
date1 = today - timedelta(days=2)
date2 = today + timedelta(days=5)
date1=date1.strftime('%Y-%m-%d')
date2=date2.strftime('%Y-%m-%d')
url=f"https://api.salvion.kr/neisApi?of=T10&sc=9290066&ac=date&sd={date1}&ed={date2}&code=all"

loader = WebBaseLoader(web_path= url)

page=loader.load()[0]
page_content= json.loads(page.page_content)
page_content= jshs.getMealMenuNeis(page_content=page_content)

documents.append( Document(
                    page_content=page_content,
                    metadata=page.metadata
                    )
                )

# 학사일정-------------------------------------------
documents.append( Document(
                        page_content=jshs.school_schedule(datetime.now().today().year),
                        metadata={'source': '홈페이지'}
                     )
                )

#  문서를 페이지로 -----------------------------------
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =1000, # 이 숫자는 매우 중요
        chunk_overlap =0, # 필요에 따라 사용
        separators=["\n\n","\n",", "], # 결국 페이지 분리를 어떻게 하느냐 가 답변의 질을 결정
        length_function =jshs.tiktoken_len
)

pages = text_splitter.split_documents(documents)

# jshs.splitter_pages_viewer(pages);quit()

vectorDB = FAISS.from_documents(pages , OpenAIEmbeddings())

today = str( datetime.now().date().today())
vectorDB_folder=f"vectorDB-faiss-jshs-{today}"

vectorDB.save_local(vectorDB_folder)

# jshs.similarity_score_viewer(vectorDB,"현재 교장은?" );quit()

from langchain.chat_models import ChatOpenAI


chain = load_qa_chain(
    ChatOpenAI(model_name="gpt-4", temperature=0),
    verbose=False
    )


query = "너의 이름은 ? "
docs = vectorDB.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)

query = "어는 학교 다녀 ? "
docs = vectorDB.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)


query = "현재 교장 선생님은? "
docs = vectorDB.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)

query = "1회 졸업 인원수 ? "
docs = vectorDB.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)


query = "학교 주소는? "
docs = vectorDB.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)


query = "오늘 날짜와 요일 말해줘 ? "
docs = vectorDB.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)

query = "몇월 달이야 ? "
docs = vectorDB.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)


query = "오늘의 점심은? "
docs = vectorDB.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)


query = f"{jshs.today_month()}  학사일정을  알려주세요. "
docs = vectorDB.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)