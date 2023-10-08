
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex
from llama_index.llms import PaLM
from llama_index import ServiceContext
from llama_index import StorageContext,load_index_from_storage
import os

from constant import GOOGLE_API_KEY

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
from IPython.display import Markdown, display

documents = SimpleDirectoryReader("../pdfs").load_data()
# print(documents)

llm = PaLM()

service_context = ServiceContext.from_defaults(
    llm=llm,
    chunk_size=1000,
    chunk_overlap=200
)

index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context
)

index.storage_context.persist()

query_engine = index.as_query_engine()
response = query_engine.query("Data Science?")

print(response)

