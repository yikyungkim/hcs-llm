import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import io
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from bs4 import BeautifulSoup
import requests



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
def extract_url_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()



# Streamlit app configuration
st.title("HCS LLM Services")
st.sidebar.title("Choose LLM Service")
option = st.sidebar.selectbox("Select an option:", ["Summarization", "Code Assistant"])


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
                st.write(summary.content)
            else:
                st.warning("Please enter text to summarize.")

    elif input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if st.button("Summarize"):
            if uploaded_file is not None:
                with st.spinner("Summarizing..."):
                    content = extract_pdf_content(uploaded_file)
                    prompt = prompt_summary.format_messages(text=content)
                    summary = llm.predict_messages(prompt)
                st.write("### Summary:")
                st.write(summary.content)
            else:
                st.warning("Please upload a PDF file.")

    elif input_method == "URL":
        url = st.text_input("Enter a URL:")
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

    
