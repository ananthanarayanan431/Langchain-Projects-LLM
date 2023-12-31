
import os
from constant import anantha_api

from constant import DeepLake,org

os.environ['OPENAI_API_KEY']=anantha_api
os.environ['ACTIVELOOP_TOKEN']=DeepLake

from langchain.document_loaders import TextLoader

text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses “generate text, images, code, videos, audio, and more from
simple natural language prompts.”

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
PaLM is a flexible system that can potentially carry out all sorts of text generation and
editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
example, or you could use it for tasks like summarizing text or even writing code.
(It’s similar to features Google also announced today for its Workspace apps like Google
Docs and Gmail.)
"""

with open("my_file.txt", "w") as file:
    file.write(text)

loader = TextLoader("my_file.txt")
docs_from_file = loader.load()

print(len(docs_from_file))

with open("my_file.txt", "w") as file:
    file.write(text)


loader = TextLoader("my_file.txt")
docs_from_file = loader.load()

# print(len(docs_from_file))

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

docs = text_splitter.split_documents(docs_from_file)

print(len(docs))

# print(len(docs))

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

from langchain.vectorstores import DeepLake

org_id = org
dataset = "langchain_course_indexers_retrievers"

dataset_path = f"hub://{org_id}/{dataset}"

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

db.add_documents(docs)

print("Successfully")

retriever = db.as_retriever()

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

qa_chain = RetrievalQA.from_chain_type(
	llm=OpenAI(model="text-davinci-003"),
	chain_type="stuff",
	retriever=retriever
)

query = "How Google plans to challenge OpenAI?"
response = qa_chain.run(query)
print(response)

print("\n\n New Output \n\n")

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

llm = OpenAI(model="text-davinci-003", temperature=0)

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
	base_compressor=compressor,
	base_retriever=retriever
)

retrieved_docs = compression_retriever.get_relevant_documents(
	"How Google plans to challenge OpenAI?"
)
print(retrieved_docs[0].page_content)