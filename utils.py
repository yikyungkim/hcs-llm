import io
import os
import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_pdf_content(uploaded_file):
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = "".join([page.extract_text() for page in pdf_reader.pages])
    return text

def extract_url_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()

def save_vector_store(vector_store, path="./vector_db"):
    vector_store.save_local(path)

def load_vector_store(path="./vector_db"):
    if os.path.exists(path):
        return FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    return None