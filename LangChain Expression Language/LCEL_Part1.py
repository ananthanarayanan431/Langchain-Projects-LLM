

from constant import openai
import os

os.environ['OPENAI_API_KEY'] = openai

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("Tell me about the Nutritional Value of {input}")

chain = LLMChain(llm=llm,prompt=prompt)
# print(chain.predict(input="Pizza"))

chain = prompt | llm
# print(chain.invoke({"input":"Chicken"}))


# print(prompt.input_schema.schema())

# print(chain.input_schema.schema())

chain = prompt | llm | StrOutputParser()
# print(chain.invoke({"input":"Lasangna"}))

prompt1 = ChatPromptTemplate.from_template(
    "Tell me 5 Jokes about {input}"
)

chain1 = prompt | llm.bind(stop=['\n']) | StrOutputParser()
# print(chain1.invoke({"input":"Pizza"}))



functions = [
    {
      "name": "joke",
      "description": "A joke",
      "parameters": {
        "type": "object",
        "properties": {
          "setup": {
            "type": "string",
            "description": "The setup for the joke"
          },
          "punchline": {
            "type": "string",
            "description": "The punchline for the joke"
          }
        },
        "required": ["setup", "punchline"]
      }
    }
  ]

chain2 = (
    prompt1
    | llm.bind(function_call={'name':'joke'}, functions=functions)
    | JsonOutputFunctionsParser()
)

print(chain2.invoke(input={'input':'cricket'}))