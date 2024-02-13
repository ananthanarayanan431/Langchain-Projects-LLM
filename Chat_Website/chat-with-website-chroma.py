import os
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = ""

# system_template = """Use the following pieces of context to answer the users question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# """
#
# messages = [
#     SystemMessagePromptTemplate.from_template(system_template),
#     HumanMessagePromptTemplate.from_template('{question}')
# ]
#
# prompt = ChatPromptTemplate.from_messages(messages)
# chain_type_kwargs = {'prompt':prompt}

def main():
    # Set the title and subtitle of the app
    st.title("Chat With Website")
    # st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')
    url = ""

    prompt = st.text_input("Ask a question query")
    if st.button("Submit Query", type="primary"):
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")

        loader = WebBaseLoader(url)
        data = loader.load()

        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=100
        )

        docs = text_splitter.split_documents(data)

        openai_embeddings = OpenAIEmbeddings()

        vectordb = Chroma.from_documents(
            documents=docs, embedding=openai_embeddings, persist_directory=DB_DIR
        )

        vectordb.persist()

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model_name="gpt-4-turbo-preview")

        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever
        )

        response = qa(prompt)
        st.write(response["result"])


if __name__ == "__main__":
    main()
