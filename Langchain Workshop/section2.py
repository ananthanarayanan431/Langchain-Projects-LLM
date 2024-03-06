
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import HNLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai.llms import OpenAI

from constant import openai
import os

os.environ['OPENAI_API_KEY'] = openai

loader = PyPDFLoader("attention is all you need.pdf")
data = loader.load()
# print(data[0])

loader = CSVLoader(file_path="job_placement.csv")
data = loader.load()
# print(data[0])

loader = HNLoader("https://news.ycombinator.com")
data = loader.load()
# print(data[0])


quote = "one Machine can do the work of fifty ordinary humans, No machine can do the" \
        "work of one extraordinary human."

ct_splitter = CharacterTextSplitter(
    separator='.',
    chunk_size=24,
    chunk_overlap=3
)

# docs = ct_splitter.split_text(quote)
# print(docs)

rc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=24,
    chunk_overlap=3,
)

# docs = rc_splitter.split_text(quote)
# print(docs)

loader = UnstructuredHTMLLoader("data.html")
data = loader.load()

rc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=24,
    chunk_overlap=3,
    separators='.',
)

# docs = rc_splitter.split_documents(data)
# print(docs)

quote = "There is a kingdom of lychee fruit that are alive and thriving in Iceland, but they feel " \
        "taken advantage of and are not fast enough for you."

splitter = RecursiveCharacterTextSplitter(
    chunk_size=40,
    chunk_overlap=10,
)

docs = splitter.split_text(quote)

embeddings = OpenAIEmbeddings(openai_api_key=openai)

vectordb = Chroma(
    persist_directory="data",
    embedding_function=embeddings
)

vectordb.persist()

docstorage = Chroma.from_texts(docs,embeddings)

qa = RetrievalQA.from_chain_type(
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct"),
    chain_type="stuff",
    retriever = docstorage.as_retriever()
)

# query = "Where do lychee fruit live?"
# print(qa.invoke(query))

quote = "There is a kingdom of lycee fruit that are alive and thriving in Iceland, but they fee" \
        "taken advantage of and are not fast enough for you."

qa1 = RetrievalQAWithSourcesChain.from_chain_type(
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct"),
    chain_type="stuff",
    retriever = docstorage.as_retriever(),
)

results = qa1({'question':'What is the primary architecture presented in the document?'},return_only_outputs=True)
print(results)
