import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os

# Import services
from summarization_service import summarization_service
from code_service import code_service
from qa_service import qa_service
from chatbot_service import chatbot_service

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# App Configuration and Styling
st.set_page_config(page_title="LLM Services", page_icon="🤖", layout="wide")
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
st.markdown("<p class='title'>LLM Services 🤖</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your AI-Powered Assistant for Summarization, Code Help, Q&A, and General Chat</p>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.header("🔄 Navigation")
services = ["Summarization", "Code Assistant", "Question and Answering", "General Chatbot"]
option = st.sidebar.radio("Select a Service", services)

# Model configuration
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
embedding_model = OpenAIEmbeddings()

# Route to the selected service
if option == "Summarization":
    summarization_service(llm)
elif option == "Code Assistant":
    code_service(llm)
elif option == "Question and Answering":
    qa_service(llm, embedding_model)
elif option == "General Chatbot":
    chatbot_service(llm)


