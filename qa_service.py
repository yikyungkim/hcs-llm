import streamlit as st
from langchain.chains import RetrievalQA
from utils import *

# pip install tiktoken
# pip install faiss-cpu

def qa_service(llm, embedding_model):
    st.subheader("❓ Question and Answering Service")
    st.info("Ask a question based on text, PDFs, or content from a URL. Choose your input method below.")

    input_method = st.radio("Input Method", ["Text", "PDF Upload", "URL"])

    # Load existing VectorStore
    vector_store = load_vector_store()

    if input_method == "Text":
        user_text = st.text_area("Enter your text:", 
                                 height=200,
                                 help="Enter the text that you want to ask questions about.")
        if st.button("Vectorize the context", key="text_vector"):
            if user_text.strip():
                with st.spinner("Vectorizing the context..."):
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    docs = splitter.split_text(user_text)
                    vector_store = FAISS.from_texts(docs, embedding_model)
                    save_vector_store(vector_store)
                    st.success("Text vectorized successfully!")

    elif input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
        if st.button("Vectorize the context", key="pdf_vector"):
            if uploaded_file:
                with st.spinner("Extracting and indexing PDF..."):
                    content = extract_pdf_content(uploaded_file)
                    document = Document(page_content=content)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    docs = splitter.split_documents([document])
                    vector_store = FAISS.from_documents(docs, embedding_model)
                    save_vector_store(vector_store)
                    st.success("PDF vectorized successfully!")
            else:
                st.warning("Please upload a PDF file and enter a question.")

    elif input_method == "URL":
        url = st.text_input("Enter a URL:")
        if st.button("Vectorize the context", key="url_vector"):
            if url.strip():
                with st.spinner("Fetching and indexing URL content..."):
                    content = extract_url_content(url)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    docs = splitter.split_text(content)
                    vector_store = FAISS.from_texts(docs, embedding_model)
                    save_vector_store(vector_store)
                    st.success("URL indexed successfully!")
            else:
                st.warning("Please enter a valid URL and a question.")

    user_question = st.text_area("Enter your question:")
    if st.button("Get Answer", key="qa"):
        if user_question and vector_store:
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            with st.spinner("Generating the answer..."):
                result = qa.run(user_question)
                st.success("### Answer")
                st.write(result)
