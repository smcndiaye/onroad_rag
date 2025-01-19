import streamlit as st
from helper import  get_conversational_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

embeddings =  GoogleGenerativeAIEmbeddings(model="models/embedding-001")

new_vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i%2 == 0:
            st.write("User: ", message.content)
        else:
            st.write("Reply: ", message.content)


def main():
    st.set_page_config("Information Retrieval")
    st.header("Information Retrieval System💁")
    user_question = st.text_input("Ask a Question about OnRoad process")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        
        user_input(user_question)
    st.session_state.conversation = get_conversational_chain(new_vector_store)


if __name__ == "__main__":
    main()