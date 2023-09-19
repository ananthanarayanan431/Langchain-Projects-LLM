
import os
from constant import openai_key
import openai
from PyPDF2 import PdfReader

os.environ['OPENAI_API_KEY']= openai_key

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch,pinecone,weaviate,FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

reader = PdfReader("Fine-Tune Your Own Llama 2 Model in a Colab Notebook _ Towards Data Science.pdf")

raw_text = ''

for i,pages in enumerate(reader.pages):
    text = pages.extract_text()
    if text:
        raw_text+=text

# print(raw_text)

textspiltter = CharacterTextSplitter(
    separator='\n',
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

texts = textspiltter.split_text(raw_text)

# print(len(texts))

embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts,embeddings)


chain = load_qa_chain(llm=OpenAI(),chain_type="stuff")

query = "Who are the authors of this article?"
doc = docsearch.similarity_search(query)

print(chain.run(input_documents=doc,question=query))

query = "LLAMA2 surpasses which models?"
doc = docsearch.similarity_search(query)
print(chain.run(input_documents=doc,question=query))