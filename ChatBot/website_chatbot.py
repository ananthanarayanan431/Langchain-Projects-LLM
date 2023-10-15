
import os
from constant import GOOGLE_API_KEY,kamalesh_oepnai

os.environ['GOOGLE_API_KEY']= 
os.environ['OPENAI_API_KEY']=

import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from HTMLTemplate import css,user_template,bot_template
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings,OpenAIEmbeddings,HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI,ChatGooglePalm
from langchain.vectorstores import FAISS

from langchain.llms import GooglePalm,OpenAI,HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

urls = ["https://medium.com/@ananthanarayanan431/the-dangers-of-ai-why-a-pioneer-of-artificial-intelligence-quit-google-a8b0cdcdac3f",
        "https://medium.com/@ananthanarayanan431/exploring-the-boundaries-of-ai-virtual-conversations-and-synthetic-companions-97332020089"]

def load_urls(url):
    loader = UnstructuredURLLoader(urls=url)
    data = loader.load()
    return data

def get_text_chunk(data):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    text_chunks = text_splitter.split_documents(data)
    return text_chunks

def get_vector_store(text_chunks):
    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(text_chunks,embedding)
    return vector_store

def get_conversational_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="ChatBot for Own Website")
    st.write(css,unsafe_allow_html=True)
    st.header("Chatbot for your website")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Enter your Question?")

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.title("LLM Chatbot using Langchain")
        st.markdown('''
        This app is an LLM powered Chatbot built using:
        - [Streamlit](https://streamlit.io/) 
        - [OpenAI](https://platform.openai.com/docs/models) LLM
        - [Lang Chain](https://python.langchain.com/)
        ''')

        st.write("Do checkout my Linkedin Profile and follow for Amazing content [Anantha Narayanan](https://www.linkedin.com/in/rananthanarayananofficial/)")

        if st.button("Start"):
            with st.spinner("Processing.."):
                data = load_urls(urls)
                text_chunk = get_text_chunk(data)

                print(len(text_chunk))
                vector_store = get_vector_store(text_chunk)

                st.session_state.conversation=get_conversational_chain(vector_store)
                st.success("Completed")



if __name__ == "__main__":
    main()
