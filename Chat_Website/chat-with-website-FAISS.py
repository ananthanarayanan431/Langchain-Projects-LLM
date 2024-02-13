from dotenv import load_dotenv

load_dotenv()


import os

os.environ["OPENAI_API_KEY"] = ""


from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
import threading

import pickle


urls = ["",
        "",
]

loaders = UnstructuredURLLoader(urls=urls)

data = loaders.load()
# print(data)

text_splitter = CharacterTextSplitter(
    separator="\n", chunk_size=1000, chunk_overlap=200
)

docs = text_splitter.split_documents(data)

# print(docs)

embeddings = OpenAIEmbeddings()

vectorstore_openai = FAISS.from_documents(docs, embeddings)

with open('faiss_store.pkl','wb') as f:
    pickle.dump(vectorstore_openai,f,protocol=pickle.HIGHEST_PROTOCOL)

with open('faiss_store.pkl','rb') as f:
    vectorstore = pickle.load(f)

llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

chain = RetrievalQAWithSourcesChain.from_llm(
    llm=llm, retriever=vectorstore_openai.as_retriever()
)

print(
    chain(
        {"question": "what is Attention model in Transformer"},
        return_only_outputs=True,
    )
)

