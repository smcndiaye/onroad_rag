import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import UnstructuredURLLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader,PyPDFLoader


load_dotenv()

def get_pdf_data(path='OnRoad datasets'):
    doc_ld = DirectoryLoader(path,glob='./*.pdf',loader_cls=PyPDFLoader)
    documents = doc_ld.load()
    return documents

def get_chunk_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    text = ""
    for d in documents:
        text += d.page_content
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def get_vector_store(text_chunks):
    embeddings =  GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2)
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain




