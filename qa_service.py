import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from utils import extract_pdf_content, extract_url_content, load_vector_store, save_vector_store


def qa_service(llm, embedding_model):
    st.subheader("❓ Question and Answering Service")
    st.info("Ask a question based on text, PDFs, or content from a URL. Choose your input method below.")

    input_method = st.radio("Input Method", ["Text", "PDF Upload", "URL"])
    question = st.text_area("Enter your question:")

    # Load existing VectorStore
    vector_store = load_vector_store()

    if input_method == "Text":
        user_text = st.text_area("Enter your text:", height=150)
        user_question = st.text_area("Enter your question:")
        if st.button("Get Answer", key="text_qa"):
            if user_text.strip():
                with st.spinner("Indexing content..."):
                    texts = [user_text]
                    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    docs = splitter.create_documents(texts)
                    vector_store = FAISS.from_documents(docs, embedding_model)
                    save_vector_store(vector_store)
                    st.success("Text indexed successfully!")

    elif input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
        user_question = st.text_area("Enter your question:")
        if st.button("Get Answer", key="pdf_qa"):
            if uploaded_file:
                with st.spinner("Extracting and indexing PDF..."):
                    content = extract_pdf_content(uploaded_file)
                    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    docs = splitter.create_documents(content)
                    vector_store = FAISS.from_documents(docs, embedding_model)
                    save_vector_store(vector_store)
                    st.success("PDF indexed successfully!")
            else:
                st.warning("Please upload a PDF file and enter a question.")

    elif input_method == "URL":
        url = st.text_input("Enter a URL:")
        user_question = st.text_area("Enter your question:")
        if st.button("Get Answer", key="url_qa"):
            if url.strip():
                with st.spinner("Fetching and indexing URL content..."):
                    content = extract_url_content(url)
                    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    docs = splitter.create_documents([content])
                    vector_store = FAISS.from_documents(docs, embedding_model)
                    save_vector_store(vector_store)
                    st.success("URL indexed successfully!")
            else:
                st.warning("Please enter a valid URL and a question.")

    if user_question and vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        with st.spinner("Generating the answer..."):
            result = qa.run(user_question)
            st.success("### Answer")
            st.write(result)
