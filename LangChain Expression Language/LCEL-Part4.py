
from operator import itemgetter
from constant import openai

import os

os.environ['OPENAI_API_KEY'] = openai

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel

llm = ChatOpenAI()

def length_function(text)->int:
    return len(text)

def _multiple_length_function(text1,text2)->int:
    return len(text1)*len(text2)

def multiple_length_function(dicti):
    return _multiple_length_function(dicti['text1'],dicti['text2'])

prompt = ChatPromptTemplate.from_template("What is {a} + {b}")

chain = {
    "a":itemgetter('foo') | RunnableLambda(length_function),
    "b": {"text1":itemgetter("foo"), "text2":itemgetter("bar")} | RunnableLambda(multiple_length_function)
} | prompt | llm | StrOutputParser()


# print(chain.invoke({"foo":"bar","bar":"gah"}))

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | model

# for s in chain.stream({"topic":"bears"}):
#     print(s.content,end=" ")

# print(chain.invoke({"topic":"bears"}))

# res = chain.batch([{"topic":"bears"},{"topic":"cats"}])
# print(res)


# async for s in chain.astream({"topic": "bears"}):
#     print(s.content, end="")
# await chain.ainvoke({"topic": "bears"})
# await chain.abatch([{"topic": "bears"}])

chain1 = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
chain2 = ChatPromptTemplate.from_template("tell me a bad joke about {topic}") | model
combined = RunnableParallel(joke=chain1, poem=chain2)

# print(chain1.invoke({"topic": "bears"}))
# print(chain2.invoke({"topic": "bears"}))
# print(combined.invoke({"topic": "bears"}))



from langchain.schema.output_parser import StrOutputParser

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a funny comedian and provide funny jokes about specific topics"),
        ("human", "Make a joke about {input}"),
    ]
)

fallback_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You tell the user that you currently are not able to make jokes since you are too tired"),
        ("human", "Make a joke about {input}"),
    ]
)

bad_llm = ChatOpenAI(model_name="gpt-fake")
bad_chain = chat_prompt | bad_llm | StrOutputParser()


llm = ChatOpenAI()
good_chain = fallback_prompt | llm


chain = bad_chain.with_fallbacks([good_chain])
print(chain.invoke({"input": "cow"}))