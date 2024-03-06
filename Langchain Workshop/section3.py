
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Chroma

from constant import openai
import os

os.environ['OPENAI_API_KEY'] = openai

model = ChatOpenAI(openai_api_key=openai,temperature=0)
prompt = ChatPromptTemplate.from_template("You're are a helpful personal assistant."
                                          "Answer the following question: {question}")

chain = prompt | model
# print(chain.invoke({'question':'Can you still have fun, Wilson?'}))

print("Streaming")
# for chunk in chain.stream({'question': "What's shaking on shakedown street?"}):
#     print(chunk.content)

print("Batching")
inputs = [{'question':"What's shaking on shakedown street?"},
          {'question':"Where is the heart of town?"}]

# results = chain.batch(inputs)
# for result in results:
#     print(result.content)

vectorstore = Chroma.from_texts(['Nothing is shaking on shakedown street.'],
                               embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

template = """Answer the question based on the context: {context}, question: {question}"""

prompt = ChatPromptTemplate.from_template(template)
chain = ({"context":retriever,"question":RunnablePassthrough()} | prompt | model | StrOutputParser())
print(chain.invoke("What is shaking on shakedown street?"))