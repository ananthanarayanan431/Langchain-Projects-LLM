
import os
from constant import openai_key


from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI

import timeit
import sys
from dotenv import load_dotenv
import pinecone

load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY','814b0486-9f63-4cc8-aea6-abaefa4b0d7b')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV', 'gcp-starter')

def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents

extracted_data=load_pdf_file(data='Data/')

#print(data)


#***Split the Data into Text Chunks****
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunk=text_splitter.split_documents(extracted_data)
    return text_chunk

text_chunks=text_split(extracted_data)
print("Length of Text Chunks", len(text_chunks))

#***Download the Embeddings from Hugging Face***
def download_hugging_face_embeddings():
    embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding

start = timeit.default_timer()
embeddings = download_hugging_face_embeddings()

query_result = embeddings.embed_query("Hello world")
print("Length", len(query_result))


#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="anantha"

#Creating Embeddings for Each of The Text Chunks
#docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

#If we already have an index we can load it like this
docsearch=Pinecone.from_existing_index(index_name, embeddings)

query = "What are Allergies"

#docs=docsearch.similarity_search(query, k=3)

#print("Result", docs)

prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="./model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})
#llm=ChatOpenAI(model_name="gpt-3.5-turbo")

qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'k': 2}),return_source_documents=True, chain_type_kwargs=chain_type_kwargs)

#query="What are Allergies"

#print("Response",qa.run(query))
while True:
    user_input=input(f"Input Prompt:")
    if user_input=='exit':
        print('Exiting')
        sys.exit()
    if user_input=='':
        continue
    result=qa({"query": user_input})
    print("Response : ", result["result"])
    print("Source Documents : ", result["source_documents"])



end=timeit.default_timer()

print(f"Time to retrieve response: {end-start}")

