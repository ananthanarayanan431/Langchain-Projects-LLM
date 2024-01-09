
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from constant import Gemini_API

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


genai.configure(api_key=Gemini_API)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embeddings)
    vector_store.save_local("Faiss_Index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['context','question'],
    )

    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("Faiss_Index",embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs,"question":user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ",response['output_text'])

def main():
    st.set_page_config("Chat with Multiple PDFS")
    st.header("Chat with PDF using Gemini")

    user_question =st.text_input("Ask a Question from the PDF files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu: ")
        pdf_docs = st.file_uploader("Upload your PDF file here..",accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done..")


main()


