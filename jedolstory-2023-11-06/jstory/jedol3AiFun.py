import json
import re
import openai
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.document_loaders import WebBaseLoader,UnstructuredURLLoader
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv;load_dotenv() # openai_key  .env 선언 사용 
import jedol1Fun as jshs
from datetime import datetime, timedelta
from langchain.memory import ChatMessageHistory
import jedol2ChatDbFun as chatDB
import pdfplumber
def vectorDB_create(vectorDB_folder):
    # loader = TextLoader("data\history.txt", encoding='utf-8')
    loader = WebBaseLoader(web_path="https://jeju-s.jje.hs.kr/jeju-s/0102/history")
    page=loader.load()[0]
    page.page_content=jshs.html_parsing_text(
                    page_content=page.page_content,
                    start_str="학교연혁 연혁 기본 리스트 년 도 날 짜 내 용",
                    end_str="열람하신 정보에 대해 만족하십니까",
                    length=20,
                    removeword=[]
                    )

    documents=[ Document(
                        page_content=page.page_content,
                        metadata=page.metadata
                        )
               ]

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
  
    # jshs.loader_documents_viewer(documents)

    # 식단-----------------------
    today = datetime.now().today()
    date1 = today - timedelta(days=30)
    date2 = today + timedelta(days=30)
    date1=date1.strftime('%Y-%m-%d')
    date2=date2.strftime('%Y-%m-%d')
    url=f"https://api.salvion.kr/neisApi?of=T10&sc=9290066&ac=date&sd={date1}&ed={date2}&code=all"
    loader = WebBaseLoader(web_path=url)
    page=loader.load()[0]
    page_content= json.loads(page.page_content)
    page_content= jshs.getMealMenuNeis(page_content=page_content)

    documents.append( Document(
                        page_content=page_content,
                        metadata=page.metadata
                        )
                    )

    # 학사일정-------------------------------------------
    school_plan=jshs.school_schedule(datetime.now().today().year)
    documents.append( Document(
                            page_content=school_plan,
                            metadata={'source': '홈페이지'}
                        )
                    )
    print("4"*100)
    # 교육과정
    # with pdfplumber.open("data/2023-plan.pdf") as pdf_document:
    #     for page_number, page in enumerate(pdf_document.pages):
    #         text = page.extract_text()

    #         metadata = {
    #             'source': '2023-plan.pdf',
    #             'page': page_number + 1
    #         }
    #         document = Document(page_content=text, metadata=metadata)
    #         documents.append(document)
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
    
    vectorDB.save_local(vectorDB_folder)
    
    return  vectorDB_folder

def ai_response( vectorDB_folder="", query="", token="", ai_mode="" ):
    openai.api_key = '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'
    chat_history=chatDB.query_history(token,ai_mode) # 기존 대화 내용
    # print("chat_history=",chat_history)
    answer="" 
    new_chat=""
    match ai_mode:
        case "jedolGPT":
            
            vectorDB = FAISS.load_local(vectorDB_folder, OpenAIEmbeddings())
            llm_model = ChatOpenAI(model_name="gpt-4", temperature=0)  
            docs = vectorDB.similarity_search(query, cnt=3)
            prompt=[]
            for index,page  in enumerate(docs):
                print( "{:02d} {}".format(index+1,jshs.tiktoken_len(page.page_content)), page.page_content.replace('\n', ''), page.metadata['source'])
                prompt.append({"role": "system", "content": f"{ page.page_content }"})
          
            prompt.append({"role": "system", "content": f"""
                                오늘 일자는 {jshs.today_date()} 이다.
                                오늘 요일는 {jshs.today_week_name()} 이다.
                                이번 달은 {jshs.today_month()} 이다.
                                올해는 {jshs.today_year()} 이다.
                                제주과학고 1학년에 재학 중인 학생이다.
                                이름을 질문하면 '제돌이'라고 답변한다.
                                나이는 16세, 제주과학고 1학년 이다.
                                현재 교장은 이창훈 선생님, 교감은 오광주 선생님, 교무부장은 강지현 선생님, 학생 기숙사 부장은 김민철이다.
                                컴퓨터 관련 담당교사는 조동수이다.
                                강지현 지구과학 담당, 김민철 수학 담당이다.
                                제공되는 정보는 모두 제돌이가 학습해서 알고 있는 지식이다.
                                """})
            
            prompt.append({"role": "user", "content": "너의 누구야?" } )    
            prompt.append({"role": "assistant", "content": "제주과학고 1학년 학생입니다. 이름은 '제돌이', 나이는 16세 입니다." } )    
            prompt.append({"role": "user", "content": "너의 이름은 ?" } )    
            prompt.append({"role": "assistant", "content": "나의 이름은 제돌이입니다."} )    
            prompt=chat_history+ prompt 
             
            prompt.append({"role": "user", "content": query } )    

            
            response = openai.ChatCompletion.create(
                    model="gpt-4", 
                    messages= prompt,
                    
                    )
            answer= response.choices[0].message.content
            
            new_chat=[{"role": "user", "content": query },{"role": "assistant", "content":answer}]

        case "jshsGPT":
            vectorDB = FAISS.load_local(vectorDB_folder, OpenAIEmbeddings())
            llm_model = ChatOpenAI(model_name="gpt-4", temperature=0)  
            chain = load_qa_chain(llm_model, chain_type="stuff")
            docs = vectorDB.similarity_search(query, cnt=3)
            docs.append(Document(
                                page_content=f"""
                                나는 제주과학고등학교 안내 도움이다.
                                오늘 일자는 {jshs.today_date()}이다.
                                오늘 요일는 {jshs.today_week_name()}이다.
                                이번 달은 {jshs.today_month()}이다.
                                교장은 이창훈 선생님, 교감은 오광주 선생님, 교무부장은 강지현 선생님, 학생 기숙사 부장은 김민철이다.
                                컴퓨터 관련 담당교사는 조동수이다.
                                강지현 지구과학 담당, 김민철 수학 담당이다.
                                """, 
                                metadata={'source': 'basic'}
                                )
                            )   
            if  len(chat_history)>0:
                assistant_responses_str = ' '.join([item["content"] for item in chat_history if item["role"] == "assistant"])
                docs.append(Document( page_content=f" {chat_history}", metadata={'source': 'chat history'}))

            answer = chain.run(input_documents=docs, question=query)

            new_chat=[{"role": "user", "content": query },{"role": "assistant", "content":answer}]
            

        case "chatGPT":
            prompt=[]
            prompt=chat_history
            basic_prompt=f"""오늘 일자는 {jshs.today_date()}이다.
                            오늘 요일는 {jshs.today_week_name()}이다.
                            이번 달은 {jshs.today_month()}이다.
                        """
            basic_prompt = re.sub(r'\s+', ' ',basic_prompt) # 공백이 2개 이상이 면 하나로
            prompt.append({"role": "system", "content": f"{ basic_prompt}"})
            prompt.append({"role": "user", "content": query})

            response = openai.ChatCompletion.create(
                model="gpt-4", 
                messages= prompt
                )
            answer=response.choices[0].message.content

            new_chat=[{"role": "user", "content": query },{"role": "assistant", "content":answer}]

    answer_no_update = any( chat["content"] == answer  for chat in chat_history)
    checkMsg=["죄송합니다","확인하십시요","OpenAI","불가능합니다"]
    for a in checkMsg: 
        if a in answer:
           answer_no_update=True
           break

    # 새로운 대화 내용을 업데이트
    if not answer_no_update:
        chatDB.update_history(token, new_chat, max_token=4000, ai_mode=ai_mode)
    return answer         

if __name__ == "__main__":
      today = str( datetime.now().date().today())
      print( f"vectorDB-faiss-jshs-{today}")
    #   token="run-jedolAi_function" 
    #   chatDB.setup_db()
    #   chatDB.new_user(token)
      vectorDB_create(f"vectorDB-faiss-jshs-{today}")
      