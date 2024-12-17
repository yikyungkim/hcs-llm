import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
<<<<<<< HEAD
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from bs4 import BeautifulSoup
import requests
from PyPDF2 import PdfReader
import io

# Helper functions
def extract_pdf_content(uploaded_file):
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = "".join([page.extract_text() for page in pdf_reader.pages])
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return " ".join(text_splitter.split_text(text))

=======

import io
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from bs4 import BeautifulSoup
import requests

os.environ['OPENAI_API_KEY'] = "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"


# # Helper function to extract content from a PDF
# def extract_pdf_content(file):
#     loader = PyPDFLoader(file)
#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     texts = text_splitter.split_documents(documents)
#     return " ".join([doc.page_content for doc in texts])

def extract_pdf_content(uploaded_file):
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    return " ".join(texts)


# Helper function to extract content from a URL
>>>>>>> e65c52c285080bab09816d770d661233b15e83d3
def extract_url_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()


<<<<<<< HEAD
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
=======

# Streamlit app configuration
st.title("HCS LLM Services")
st.sidebar.title("Choose LLM Service")
services = ["Summarization", "Code Assistant", "General Chatbot"]
option = st.sidebar.selectbox("Select an option:", services)


# Initialize OpenAI model with LangChain
llm = ChatOpenAI(temperature=0.7)


# Define the summarization prompt template
prompt_summary = ChatPromptTemplate.from_messages([
    ("system", """
        You are a professional text summarizer. 
        Summarize the text into three categories: Main Idea, Key Details, and Additional Insights. 
        Ensure that your summary is accurate and well-structured. 
        Please respond in Korean
    """),
    ("human", """
        {text}
        
        Summary:
        - Main Idea:
        - Key Points:
        - Insights:
    """),
])

# Define the coding assistant prompt template
prompt_code = ChatPromptTemplate.from_messages([
    ("system", """
        You are a coding assistant specialized in helping with programming problems, 
        debugging code, and explaining technical concepts. Always provide accurate and 
        easy-to-understand answers.
    """),
    ("human", "{question}"),
])

if option == "Summarization":
    st.header("Summarization Service")
    st.write("Provide input as text, upload a PDF, or paste a URL to summarize the content.")

    input_method = st.selectbox("Input Method", ["Text", "PDF Upload", "URL"])

    if input_method == "Text":
        user_text = st.text_area("Enter your text here:",
                                 height=200,
                                 help="Enter the text that you want to summarize.")
        if st.button("Summarize"):
            if user_text.strip():
                with st.spinner("Summarizing..."):
                    prompt = prompt_summary.format_messages(text=user_text)
                    summary = llm.predict_messages(prompt)
                st.write("### Summary:")
>>>>>>> e65c52c285080bab09816d770d661233b15e83d3
                st.write(summary.content)
            else:
                st.warning("Please enter text to summarize.")

    elif input_method == "PDF Upload":
<<<<<<< HEAD
        uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
        if st.button("Summarize", key="pdf_summarize"):
            if uploaded_file:
                with st.spinner("Extracting content and summarizing..."):
                    content = extract_pdf_content(uploaded_file)
                    prompt = prompt_summary.format_messages(text=content)
                    summary = llm.predict_messages(prompt)
                st.success("### Summary")
=======
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if st.button("Summarize"):
            if uploaded_file is not None:
                with st.spinner("Summarizing..."):
                    content = extract_pdf_content(uploaded_file)
                    prompt = prompt_summary.format_messages(text=content)
                    summary = llm.predict_messages(prompt)
                st.write("### Summary:")
>>>>>>> e65c52c285080bab09816d770d661233b15e83d3
                st.write(summary.content)
            else:
                st.warning("Please upload a PDF file.")

    elif input_method == "URL":
        url = st.text_input("Enter a URL:")
<<<<<<< HEAD
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

=======
        if st.button("Summarize"):
            if url.strip():
                with st.spinner("Summarizing..."):
                    content = extract_url_content(url)
                    prompt = prompt_summary.format_messages(text=content)
                    summary = llm.predict_messages(prompt)
                st.subheader("Summary")
                st.write(summary)
            else:
                st.warning("Please provide a valid URL.")


elif option=="Code Assistant":
    st.write("## Ask the Code Assistant")
    user_question = st.text_area(
        "Describe your programming question or issue here:",
        height=150,
        help="You can ask about coding, debugging, or technical concepts.")

    if st.button("Get Answer"):
        if user_question.strip():
            with st.spinner("Generating answer..."):
                prompt = prompt_code.format_messages(question=user_question)
                response = llm.predict_messages(prompt)
            st.write("### Assistant's Response:")
            st.code(response.content, language="python")  # Assuming Python output; adjust as needed
        else:
            st.warning("Please enter a question or issue.")

    
elif option=="General Chatbot":
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)
    st.write("## What can I help with?")
    
>>>>>>> e65c52c285080bab09816d770d661233b15e83d3
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

<<<<<<< HEAD
    if user_input := st.chat_input("Ask me anything..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})
        response = conversation.run(input=user_input)
        with st.chat_message("assistant"):
            st.markdown(response)
=======
    if user_input := st.chat_input("Enter message here"):
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state["messages"].append({"role": "user", "content": user_input})
        response = conversation.run(input=user_input)
        
        with st.chat_message("assistant"):
            st.markdown(response)

>>>>>>> e65c52c285080bab09816d770d661233b15e83d3
        st.session_state["messages"].append({"role": "assistant", "content": response})
