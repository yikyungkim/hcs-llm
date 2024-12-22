# pip install tiktoken
# pip install faiss-cpu

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests
import io
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Helper functions
def extract_pdf_content(uploaded_file):
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = "".join([page.extract_text() for page in pdf_reader.pages])
    return text
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # return " ".join(text_splitter.split_text(text))

def extract_url_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()

# Save and Load VectorStore
def save_vector_store(vector_store, path="vector_store"):
    vector_store.save_local(path)

def load_vector_store(path="vector_store"):
    if os.path.exists(path):
        return FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    return None

# App Configuration and Styling
st.set_page_config(page_title="HCS LLM Services", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
        .title {
            font-size: 2.5em;
            font-weight: bold;
            color: #4A90E2;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 1.5em;
            color: #333333;
            text-align: center;
        }
        .stButton>button {
            width: 100%;
            margin-top: 10px;
        }
        .reportview-container .markdown-text-container {
            font-family: 'Arial', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown("<p class='title'>HCS LLM Services 🤖</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your AI-Powered Assistant for Summarization, Code Help, Q&A, and General Chat</p>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.header("🔄 Navigation")
services = ["Summarization", "Code Assistant", "Question and Answering", "General Chatbot"]
option = st.sidebar.radio("Select a Service", services)

# Model configuration settings
st.sidebar.header("⚙️ Model Configuration")
model_type = st.sidebar.selectbox("Model Type:", ["gpt-4", "gpt-3.5-turbo", "gpt-4o"], index=0)
response_type = st.sidebar.selectbox("Select Response Type:", ["Creative", "Balanced", "Accurate"], index=1)

# Pre-configured settings based on response type
if response_type == "Creative":
    temperature, top_p, p_penalty, f_penalty, max_tokens = 0.8, 0.9, 0.1, 0.1, 500
elif response_type == "Balanced":
    temperature, top_p, p_penalty, f_penalty, max_tokens = 0.5, 0.8, 0.2, 0.3, 500
elif response_type == "Accurate":
    temperature, top_p, p_penalty, f_penalty, max_tokens = 0.2, 0.7, 0.5, 0.5, 500

# Custom Configuration
show_custom_settings = st.sidebar.checkbox("Show Custom Model Settings")

if show_custom_settings:
    st.sidebar.subheader("Custom Model Settings")
    temperature = st.sidebar.slider("Temperature (for randomness):", 0.0, 1.0, temperature, 0.1)
    top_p = st.sidebar.slider("Top-p (nucleus sampling):", 0.1, 1.0, top_p, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens (output length):", 50, 2000, 500, 50)
    p_penalty = st.sidebar.slider("Presence Penalty:", 0.1, 1.0, p_penalty, 0.1)
    f_penalty = st.sidebar.slider("Frequency Penalty:", 0.1, 1.0, f_penalty, 0.1)

# Update the model based on user selection
llm = ChatOpenAI(model=model_type, temperature=temperature, max_tokens=max_tokens, top_p=top_p, presence_penalty=p_penalty, frequency_penalty=f_penalty)
embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen2-72B-Instruct")
# text = "Your input text here"
# embedded_text = embeddings.embed_query(text)

# Prompt Templates
prompt_summary = ChatPromptTemplate.from_messages([
    ("system", "You are a professional text summarizer. Summarize the text into three categories: Main Idea, Key Points, and Insights. Please respond in Korean."),
    ("human", "{text}\n\nSummary:\n- Main Idea:\n- Key Points:\n- Insights:")
])

prompt_code = ChatPromptTemplate.from_messages([
    ("system", "You are a coding assistant specialized in helping with programming problems, debugging, and explaining technical concepts."),
    ("human", "{question}")
])

prompt_qa = ChatPromptTemplate.from_messages([
    ("system", """You are a highly intelligent assistant specialized in answering questions based on provided context.  
Analyze the given information carefully and provide clear, concise, and accurate responses.  
If the context does not contain enough information, state that explicitly.  
Structure your answer as follows:  

1. **Direct Answer**: Provide a one-sentence summary if possible.  
2. **Detailed Explanation**: Expand with relevant details from the context.  
3. **References**: Highlight specific sections of the context, if applicable.  

Context: {context}  
Question: {question}  
Answer:""")])


# Main Content
if option == "Summarization":
    st.subheader("🕵️ Summarization Service")
    st.info("Summarize text, PDFs, or content from a URL. Choose your input method below.")

    input_method = st.radio("Input Method", ["Text", "PDF Upload", "URL"])

    if input_method == "Text":
        user_text = st.text_area("Enter your text:", height=150)
        if st.button("Summarize", key="text_summarize"):
            if user_text.strip():
                with st.spinner("Summarizing text..."):
                    prompt = prompt_summary.format_messages(text=user_text)
                    summary = llm.predict_messages(prompt)
                st.success("### Summary")
                st.write(summary.content)
            else:
                st.warning("Please enter text to summarize.")

    elif input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
        if st.button("Summarize", key="pdf_summarize"):
            if uploaded_file:
                with st.spinner("Extracting content and summarizing..."):
                    content = extract_pdf_content(uploaded_file)
                    prompt = prompt_summary.format_messages(text=content)
                    summary = llm.predict_messages(prompt)
                st.success("### Summary")
                st.write(summary.content)
            else:
                st.warning("Please upload a PDF file.")

    elif input_method == "URL":
        url = st.text_input("Enter a URL:")
        if st.button("Summarize", key="url_summarize"):
            if url.strip():
                with st.spinner("Fetching content and summarizing..."):
                    content = extract_url_content(url)
                    prompt = prompt_summary.format_messages(text=content)
                    summary = llm.predict_messages(prompt)
                st.success("### Summary")
                st.write(summary.content)
            else:
                st.warning("Please enter a valid URL.")

elif option == "Code Assistant":
    st.subheader("🛠️ Code Assistant")
    st.info("Describe your coding problem, and I will assist with solutions, debugging, or explanations.")

    user_question = st.text_area("Enter your coding question:", height=150)
    if st.button("Get Solution", key="code_solution"):
        if user_question.strip():
            with st.spinner("Generating solution..."):
                prompt = prompt_code.format_messages(question=user_question)
                response = llm.predict_messages(prompt)
            st.success("### Assistant's Response")
            st.code(response.content, language="python")
        else:
            st.warning("Please enter a coding question.")

elif option == "Question and Answering":
    st.subheader("❓ Question and Answering Service")
    st.info("Ask a question based on text, PDFs, or content from a URL. Choose your input method below.")

    input_method = st.radio("Input Method", ["Text", "PDF Upload", "URL"])    
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


elif option == "General Chatbot":
    st.subheader("🧑‍💬 General Chatbot")
    st.info("Chat with the assistant for any questions or help.")

    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory()

    memory = st.session_state["memory"]
    conversation = ConversationChain(llm=llm, memory=memory)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask me anything..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})

        response = conversation.run(input=user_input)
        
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
