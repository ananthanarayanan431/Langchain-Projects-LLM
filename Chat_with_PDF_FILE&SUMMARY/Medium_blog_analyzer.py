
from langchain.embeddings import OpenAIEmbeddings,GooglePalmEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
import pinecone
from langchain.llms import OpenAI

import os
from third_party.constant import kamalesh_oepnai

os.environ['OPENAI_API_KEY'] = kamalesh_oepnai

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY',"814b0486-9f63-4cc8-aea6-abaefa4b0d7b")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV","gcp-starter")


pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)

index_name="anantha"

loader = PyPDFLoader("Vector.pdf")
document = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
texts = text_splitter.split_documents(document)
# print(len(texts))

embedding = OpenAIEmbeddings()
docsearch = Pinecone.from_documents(
    texts,
    embedding=embedding,
    index_name="anantha"
)

qa = RetrievalQA.from_chain_type(
    llm = OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

query="What is Vector DataBases give in answer in 20 words for a Beginner"
result=qa({'query':query})

print(result)