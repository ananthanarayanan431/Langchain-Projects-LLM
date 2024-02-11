from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.pinecone import Pinecone as PineconeLangchain

from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "anantha"


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docsearch = PineconeLangchain.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )

    chat = ChatGoogleGenerativeAI(
        model="gemini-pro",
        convert_system_message_to_human=True,
        verbose=True,
        temperature=True,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    return qa.invoke({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="what is Langchain")["answer"])
