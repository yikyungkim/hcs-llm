import streamlit as st
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, PDFMinerLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
import PyPDF2  # Using PyPDF2 for PDF parsing
from langchain.schema import Document

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Function to summarize using load_summarize_chain
def summarize_document(text, method='stuff'):
    # Split the text into smaller chunks if it's too large
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    
    # Use load_summarize_chain to get the summarization chain
    chain = load_summarize_chain(llm, chain_type=method)  # method can be 'stuff', 'map_reduce', or 'refine'

    # Summarize the text
    summary = chain.run(docs)
    return summary

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_file):
    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    
    # Loop through each page and extract text
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    
    return text

# Function to scrape content from a URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

# Main Streamlit interface
st.title("Document Summarizer")
st.write("Upload a PDF, paste a URL, or enter text to summarize")

input_type = st.radio("Select input type:", ['Text', 'PDF', 'URL'])

if input_type == 'Text':
    user_text = st.text_area("Enter the text to summarize", height=300)
    method = st.selectbox("Select summarization method", ['stuff', 'map_reduce', 'refine'])

    if st.button("Summarize"):
        if user_text:
            summary = summarize_document(user_text, method)
            st.subheader("Original Text:")
            st.write(user_text[:500] + "...")
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.error("Please enter some text to summarize.")

elif input_type == 'PDF':
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")

    if pdf_file:
        method = st.selectbox("Select summarization method", ['stuff', 'map_reduce', 'refine'])
        if st.button("Summarize PDF"):
            text = extract_text_from_pdf(pdf_file)
            summary = summarize_document(text, method)
            st.subheader("Original PDF Text:")
            st.write(text[:500] + "...")
            st.subheader("Summary:")
            st.write(summary)

elif input_type == 'URL':
    url = st.text_input("Enter URL to summarize")

    if url:
        method = st.selectbox("Select summarization method", ['stuff', 'map_reduce', 'refine'])
        if st.button("Summarize URL"):
            text = extract_text_from_url(url)
            summary = summarize_document(text, method)
            st.subheader("Original Web Content:")
            st.write(text[:500] + "...")
            st.subheader("Summary:")
            st.write(summary)
