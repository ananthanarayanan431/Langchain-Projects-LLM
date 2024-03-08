
from operator import itemgetter
from constant import openai

import os

os.environ['OPENAI_API_KEY'] = openai

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_openai.chat_models import ChatOpenAI

from langchain.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

question = {'bla':'test','x':'hi'}
# print(itemgetter('bla'))

get_bla = itemgetter('bla')
# print(get_bla(question))

llm = ChatOpenAI()

vectorstore = Chroma.from_texts(["Cats are typically 9.1 kg in weight.",
                                 "Cats have retractable claws.",
                                 "A group of cats is called a clowder.",
                                 "Cats can rotate their ears 180 degrees.",
                                 "The world's oldest cat lived to be 38 years old."],
                                embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = {
    "context": (lambda x :x['question']) | retriever,
    "question": (lambda  x: x['question']),
    'language': (lambda x:x['language'])
} | prompt | llm | StrOutputParser()

# print(chain.invoke({"question":"How old is the lodest cat?", "language":"Hindi"}))


template = """
Turn the following user input into search query for a search engine:
{input}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm | StrOutputParser() | search

print(chain.invoke({"input":"What's the name of theoldest cat?"}))
