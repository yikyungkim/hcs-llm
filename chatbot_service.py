import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


def chatbot_service(llm):
    st.subheader("🧑‍💬 General Chatbot")
    st.info("Chat with the assistant for any questions or help.")

    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(return_messages=True)

    memory = st.session_state["memory"]
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

        with st.chat_message("assistant"):
            response = conversation.run(input=user_input)
            st.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
