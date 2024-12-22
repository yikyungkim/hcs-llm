import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils import extract_pdf_content, extract_url_content

def summarization_service(llm):
    st.subheader("🕵️ Summarization Service")
    st.info("Summarize text, PDFs, or content from a URL. Choose your input method below.")

    prompt_summary = ChatPromptTemplate.from_messages([
        ("system", "You are a professional text summarizer. Summarize the text into three categories: Main Idea, Key Points, and Insights."),
        ("human", "{text}\n\nSummary:\n- Main Idea:\n- Key Points:\n- Insights:")
    ])

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