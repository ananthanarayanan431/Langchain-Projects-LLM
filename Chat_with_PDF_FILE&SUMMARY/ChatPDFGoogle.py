
import os
from constant import GOOGLE_API_KEY

os.environ['GOOGLE_API_KEY']= GOOGLE_API_KEY

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
import google.generativeai as palm

from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain #comes with memory component
from langchain.memory import ConversationBufferMemory

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text() is not None:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks,embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vector_store.as_retriever(),memory=memory)
    return conversation_chain

def user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chatHistory = response['chat_history']
    for i,message in enumerate(st.session_state.chatHistory):
        if i%2==0:
            st.write("Human : ",message.content)
        else:
            st.write("Bot : ",message.content)
        # st.write(i,message)

def main():
    st.set_page_config("Chat with Multiple PDF's")
    st.header("Chat with Multiple PDF")
    user_question = st.text_input("Ask a Question from your PDFs")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        pdfs = st.file_uploader("Upload your PDF files here",accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdfs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Done")

if __name__ == "__main__":
    main()





