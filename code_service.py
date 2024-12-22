import streamlit as st
from langchain.prompts import ChatPromptTemplate


def code_service(llm):
    st.subheader("🛠️ Code Assistant")
    st.info("Describe your coding problem, and I will assist with solutions, debugging, or explanations.")

    prompt_code = ChatPromptTemplate.from_messages([
        ("system", "You are a coding assistant specialized in helping with programming problems, debugging, and explaining technical concepts."),
        ("human", "{question}")
    ])

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
