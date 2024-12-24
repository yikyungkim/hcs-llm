import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import *

from langchain.chains.summarize import load_summarize_chain

def summarize_text(text, llm):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]

    # prompt_template = """
    #     You are a professional text summarizer. Summarize the following text in Korean:
    #     {text}
        
    #     Provide the summary in the following format:
    #     Main Idea: [The central concept or argument]
    #     Key Points:
    #     - [Key point 1]
    #     - [Key point 2]
    #     - [Key point 3]
    #     Insights: [Any additional insights or implications]
    #     """

    # PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])    
    # chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary


def summarization_service(llm):
    st.subheader("🕵️ Summarization Service")
    st.info("Summarize text, PDFs, or content from a URL. Choose your input method below.")

    input_method = st.radio("Input Method", ["Text", "PDF Upload", "URL"])

    if input_method == "Text":
        content = st.text_area("Enter your text:", height=200)
        if st.button("Summarize", key="text_summarize"):
            if content.strip():
                with st.spinner("Summarizing text..."):
                    summary = summarize_text(content, llm)
                    # prompt = prompt_summary.format_messages(text=content)
                    # summary = llm.predict_messages(prompt)
                st.success("### Summary")
                st.write(summary)
            else:
                st.warning("Please enter text to summarize.")

    elif input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
        if st.button("Summarize", key="pdf_summarize"):
            if uploaded_file:
                with st.spinner("Extracting content and summarizing..."):
                    content = extract_pdf_content(uploaded_file)
                    summary = summarize_text(content, llm)
                    # prompt = prompt_summary.format_messages(text=content)
                    # summary = llm.predict_messages(prompt)
                st.success("### Summary")
                st.write(summary)
            else:
                st.warning("Please upload a PDF file.")


    elif input_method == "URL":
        url = st.text_input("Enter a URL:")
        if st.button("Summarize", key="url_summarize"):
            if url.strip():
                with st.spinner("Fetching content and summarizing..."):
                    content = extract_url_content(url)
                    summary = summarize_text(content, llm)
                    # prompt = prompt_summary.format_messages(text=content)
                    # summary = llm.predict_messages(prompt)
                st.success("### Summary")
                st.write(summary)
            else:
                st.warning("Please enter a valid URL.")
