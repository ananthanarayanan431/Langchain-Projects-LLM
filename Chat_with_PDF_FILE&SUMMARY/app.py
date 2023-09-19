
from constant import openai_key
import os
import openai

import streamlit as st
import pickle
from PyPDF2 import PdfReader

os.environ['OPENAI_API_KEY']= openai_key

from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    st.header("Chat with your PDF Data")
    st.sidebar.title("LLM ChatPDF APP using LangChain")
    st.sidebar.markdown('''
    This is LLM Powered ChatBot and Built Using:
    - [Streamlit] (https://streamlit.io/)
    - [Langchain] (https://www.langchain.com/)
    - [OpenAI] (https://openai.com/)
    ''')

    st.sidebar.write("Do checkout my Linkedin Profile and follow for Amazing content [Anantha Narayanan] (https://www.linkedin.com/in/rananthanarayananofficial/)")

    pdf = st.file_uploader("Upload a PDF file",type="pdf")

    if pdf is not None:
        reader = PdfReader(pdf)
        raw_text = ""
        for i,pages in enumerate(reader.pages):
            text = pages.extract_text()
            if text is not None:
                raw_text += text

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunk = text_splitter.split_text(text=raw_text)

        # st.write(chunk[0])

        store_name = pdf.name[:-4]
        st.write(store_name)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vectorstore = pickle.load(f)
            st.write("Embedding Loaded from Disk")
        else:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunk,embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vectorstore,f)
            st.write("Embedding Created in Disk")

        query = st.text_input("Ask a Question in the PDF")
        if query:
            docs = vectorstore.similarity_search(query=query)
            chain = load_qa_chain(llm=OpenAI(),chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,question=query)
                print(cb)
            st.write(response)

if __name__=="__main__":
    main()
