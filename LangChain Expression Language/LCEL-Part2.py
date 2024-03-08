
from constant import openai
import os

os.environ['OPENAI_API_KEY'] = openai

from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

from operator import itemgetter


llm = ChatOpenAI()

vectorstore = Chroma.from_texts(
    [
        "Cats are typically 9.1 kg in weight.",
        "Cats have retractable claws.",
        "A group of cats is called a clowder.",
        "Cats can rotate their ears 180 degrees.",
        "The world's oldest cat lived to be 38 years old."
    ],
    embedding = OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {'context':retriever, "question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# print(chain.invoke("How old is the oldest cat?"))

question = {'bla':'test','x':'hi'}
print(itemgetter('bla'))

get_bla = itemgetter('bla')
print(get_bla(question))