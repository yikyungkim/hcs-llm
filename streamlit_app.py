import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from bs4 import BeautifulSoup
import requests
from PyPDF2 import PdfReader
import io
import os
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Helper functions
def extract_pdf_content(uploaded_file):
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = "".join([page.extract_text() for page in pdf_reader.pages])
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return " ".join(text_splitter.split_text(text))

def extract_url_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()


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
st.markdown("<p class='subtitle'>Your AI-Powered Assistant for Summarization, Code Help, and General Chat</p>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.header("🔄 Navigation")
services = ["Summarization", "Code Assistant", "General Chatbot"]
option = st.sidebar.radio("Select a Service", services)

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Prompt Templates
prompt_summary = ChatPromptTemplate.from_messages([
    ("system", "You are a professional text summarizer. Summarize the text into three categories: Main Idea, Key Points, and Insights. Please respond in Korean."),
    ("human", "{text}\n\nSummary:\n- Main Idea:\n- Key Points:\n- Insights:")
])

prompt_code = ChatPromptTemplate.from_messages([
    ("system", "You are a coding assistant specialized in helping with programming problems, debugging, and explaining technical concepts."),
    ("human", "{question}")
])

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

elif option == "General Chatbot":
    st.subheader("🧑‍💬 General Chatbot")
    st.info("Chat with the assistant for any questions or help.")
    memory = ConversationBufferMemory()
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
