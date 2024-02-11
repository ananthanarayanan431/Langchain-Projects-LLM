import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeLangchain
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("pinecone_api_key"))

INDEX_NAME = "anantha"


def ingest_docs():
    loader = ReadTheDocsLoader(
        path=r"langchain-docs/api.python.langchain.com/en/latest", encoding="ISO-8859-1"
    )

    raw_document = loader.load()
    print(f"Loaded {len(raw_document)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

    documents = text_splitter.split_documents(raw_document)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print(f"Gooing to add {len(documents)} to pinecone")
    PineconeLangchain.from_documents(
        documents,
        embeddings,
        index_name=INDEX_NAME,
    )

    print("***Loading to VectorStore Done!")


if __name__ == "__main__":
    ingest_docs()
