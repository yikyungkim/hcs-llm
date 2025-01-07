import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain

from utils import *


def summarization_service(llm):
    st.subheader("🕵️ Summarization Service")
    st.info("Summarize text, PDFs, or content from a URL. Choose your input method below.")

    # 입력 데이터 선택
    input_method = st.radio("Input Method", ["Text", "PDF Upload", "URL"])

    content=""
    if input_method == "Text":
        content = st.text_area("Enter your text:", 
                                 height=200,
                                 help="Enter the text that you want to summarize.")

    elif input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
        if uploaded_file:
            content = extract_pdf_content(uploaded_file)

    elif input_method == "URL":
        url = st.text_input("Enter a URL:")
        if url.strip():
            content = extract_url_content(url)

    prompt_summary = ChatPromptTemplate.from_template(
        """
        You are a professional summarization assistant. Your task is to summarize the provided text in Korean while maintaining accuracy and clarity. 

        Instructions: 
        1. Provide a brief summary of the main point in 1-2 sentences.  
        2. List key details and highlights using bullet points.  
        3. Offer insights or takeaways based on the content.  

        Output Format (in Korean): 
        1. **핵심 요약:**  
        - [Write the main point here in 1-2 sentences.]  

        2. **주요 내용:**  
        - [Bullet point 1]  
        - [Bullet point 2]  
        - [Bullet point 3]  

        3. **인사이트 및 시사점 (Insights and Takeaways):**  
        - [Insight 1]  
        - [Insight 2]  
        - [Insight 3]  

        Input Text: {text}
        """
    )

    # 텍스트 분할 및 요약
    if st.button("Summarize", key="summarize") and content:
        with st.spinner("문서를 요약하는 중입니다..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            chunks = splitter.split_text(content)

            # Generate summaries for each chunk
            summaries = []
            for chunk in chunks:
                prompt = prompt_summary.format(text=chunk)
                response = llm.predict(prompt)
                summaries.append(response)
            st.success("### AI Summarization Assistant's Response")
            for idx, summary in enumerate(summaries):
                st.write(f"### Part {idx+1}")
                st.write(summary)


# Advanced summarization service
def summarization_service_advanced(llm):
    st.subheader("🕵️ Summarization Service")
    st.info("Summarize text, PDFs, or content from a URL. Choose your input method below.")

    # 입력 데이터 선택
    input_method = st.radio("Input Method", ["Text", "PDF Upload", "URL"])

    if input_method == "Text":
        content = st.text_area("Enter your text:", 
                                 height=200,
                                 help="Enter the text that you want to summarize.")

    elif input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
        if uploaded_file:
            content = extract_pdf_content(uploaded_file)

    elif input_method == "URL":
        url = st.text_input("Enter a URL:")
        if url.strip():
            content = extract_url_content(url)

    # 텍스트 분할 및 요약
    if st.button("Summarize", key="summarize") and content:
        with st.spinner("문서를 요약하는 중입니다..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            document = Document(page_content=content)
            docs = splitter.split_documents([document])

            prompt_template = """
                You are a professional text summarizer. Summarize the following text in Korean:
                {text}
                
                Provide the summary in the following format:
                Main Idea: [The central concept or argument]
                Key Points:
                - [Key point 1]
                - [Key point 2]
                - [Key point 3]
                Insights: [Any additional insights or implications]
            """
            prompt = PromptTemplate.from_template(prompt_template)

            refine_template = (
                "Your job is to produce a final summary\n"
                "We have provided an existing summary up to a certain point: {existing_answer}\n"
                "We have the opportunity to refine the existing summary"
                "(only if needed) with some more context below.\n"
                "------------\n"
                "{text}\n"
                "------------\n"
                "Given the new context, refine the original summary in Korean"
                "If the context isn't useful, return the original summary."
            )
            refine_prompt = PromptTemplate(template=refine_template)   

            chain = load_summarize_chain(
                llm=llm,
                chain_type="refine",
                question_prompt=prompt,
                refine_prompt=refine_prompt,
                return_intermediate_steps=True,
                input_key="input_documents",
                output_key="output_text",)
            # summary = chain.run(docs)
            summary = chain({"input_documents": docs}, return_only_outputs=True)
            st.success("### Summary")
            st.write(summary)
    else:
        st.warning("Please enter text to summarize.")
