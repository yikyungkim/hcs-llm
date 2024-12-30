import streamlit as st
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS

from langchain.callbacks.base import BaseCallbackHandler
from functools import lru_cache

from utils import *

# pip install tiktoken
# pip install faiss-cpu

# def qa_service(llm, embedding_model):
#     st.subheader("❓ Question and Answering Service")
#     st.info("Ask a question based on text, PDFs, or content from a URL. Choose your input method below.")

#     input_method = st.radio("Input Method", ["Text", "PDF Upload", "URL"])

#     # Load existing VectorStore
#     vector_store = load_vector_store()

#     if input_method == "Text":
#         user_text = st.text_area("Enter your text:", 
#                                  height=200,
#                                  help="Enter the text that you want to ask questions about.")
#         if st.button("Vectorize the context", key="text_vector"):
#             if user_text.strip():
#                 with st.spinner("Vectorizing the context..."):
#                     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#                     docs = splitter.split_text(user_text)
#                     vector_store = FAISS.from_texts(docs, embedding_model)
#                     save_vector_store(vector_store)
#                     st.success("Text vectorized successfully!")

#     elif input_method == "PDF Upload":
#         uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
#         if st.button("Vectorize the context", key="pdf_vector"):
#             if uploaded_file:
#                 with st.spinner("Extracting and indexing PDF..."):
#                     content = extract_pdf_content(uploaded_file)
#                     document = Document(page_content=content)
#                     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#                     docs = splitter.split_documents([document])
#                     vector_store = FAISS.from_documents(docs, embedding_model)
#                     save_vector_store(vector_store)
#                     st.success("PDF vectorized successfully!")
#             else:
#                 st.warning("Please upload a PDF file and enter a question.")

#     elif input_method == "URL":
#         url = st.text_input("Enter a URL:")
#         if st.button("Vectorize the context", key="url_vector"):
#             if url.strip():
#                 with st.spinner("Fetching and indexing URL content..."):
#                     content = extract_url_content(url)
#                     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#                     docs = splitter.split_text(content)
#                     vector_store = FAISS.from_texts(docs, embedding_model)
#                     save_vector_store(vector_store)
#                     st.success("URL indexed successfully!")
#             else:
#                 st.warning("Please enter a valid URL and a question.")

    # user_question = st.text_area("Enter your question:")
    # if st.button("Get Answer", key="qa"):
    #     if user_question and vector_store:
    #         retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    #         qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    #         with st.spinner("Generating the answer..."):
    #             result = qa.run(user_question)
    #             st.success("### Answer")
    #             st.write(result)


def qa_service(llm, embedding_model):
    st.subheader("❓ Question and Answering Service")
    st.info("Ask a question based on text, PDFs, or content from a URL. Choose your input method below.")

    # 대화 메모리 설정
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = st.session_state["memory"]

    # 입력 데이터 선택
    input_method = st.radio("Input Method", ["Text", "PDF Upload", "URL"])

    if input_method == "Text":
        content = st.text_area("Enter your text:", 
                                 height=200,
                                 help="Enter the text that you want to ask questions about.")

    elif input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
        if uploaded_file:
            content = extract_pdf_content(uploaded_file)

    elif input_method == "URL":
        url = st.text_input("Enter a URL:")
        if url.strip():
            content = extract_url_content(url)

    # 텍스트 분할 및 휘발성 Vector DB 생성
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    if st.button("업로드", key="upload") and content:
        with st.spinner("문서를 업로드 하는 중입니다..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            if input_method in ["Text", "URL"]:
                docs = splitter.split_text(content)
                st.session_state.vector_db = FAISS.from_texts(docs, embedding_model)        
            elif input_method == "PDF Upload":
                document = Document(page_content=content)
                docs = splitter.split_documents([document])
                st.session_state.vector_db = FAISS.from_documents(docs, embedding_model)
            st.success("로딩이 완료되었습니다.")
            st.session_state["messages"] = []

    if st.session_state.vector_db:
        # 관련 문서 검색
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})

        # 대화형 QA 체인 설정
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
        # qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # 대화 메시지 표시
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 대화 메시지 업데이트
        if user_input := st.chat_input("업로드 된 문서에 대한 질문을 입력하세요..."):
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state["messages"].append({"role": "user", "content": user_input})

            # 답변 생성
            with st.chat_message("assistant"):
                with st.spinner("답변을 생성중입니다..."):
                    response = qa_chain.run(user_input)
                    st.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})

