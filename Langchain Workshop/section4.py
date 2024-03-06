

from constant import openai
import os

from langchain.agents import tool
from langchain.agents import AgentType,Tool,initialize_agent
from langchain_openai.llms import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import format_tool_to_openai_function
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = openai

from langchain.evaluation import Criteria
from langchain.evaluation import load_evaluator
from langchain.evaluation import QAEvalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

@tool
def finacial_report(company_name:str)->str:
    """Generate a finacial report for a company that calualates net income"""

    revenue = 10_00_000
    expenses = 500_000
    net_income = revenue-expenses

    report = f"Extermely basic finacial report inculding net income for {company_name}\n"
    report += f"Revenue : ${revenue}\n"
    report += f"Expenses: ${expenses}\n"
    report += f"Net Income: ${net_income}\n"

    return report

def divisible_by_five(n:int)->int:
    """Calculate the number of times an input is divisible by five"""
    return n//5

tools = [
    Tool(
        name="FinaceReport",
        func=finacial_report,
        description="Use this for running a finacial report for net income",
    )
]

llm = OpenAI(temperature=0,openai_api_key=openai)
agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

question = "Run a finacial report for that calculates net income for Hooli"

# print(agent.run(question))

divisible_tool = StructuredTool.from_function(divisible_by_five)

agent = initialize_agent(
    tools=[divisible_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# result = divisible_tool.func(n=25)
# print(result)


class FinacialReportDescription(BaseModel):
    query:str = Field(description='generate a finacial report using net income')

@tool(args_schema=FinacialReportDescription)
def finacial_report_oai(company_name:str)->str:
    """Hello"""
    [...]

# print(format_tool_to_openai_function(finacial_report_oai))


class CallingItBack(BaseCallbackHandler):
    def on_llm_start(self,serilized,prompts,invocation_params,**kwargs):
        print(prompts)
        print(serilized)
        print(invocation_params['model_name'])
        print(invocation_params['temperature'])

    def on_llm_new_token(self,token:str,**kwargs)-> None:
        print(repr(token))

llm = OpenAI(model_name="gpt-3.5-turbo-instruct",streaming=True)

prompt_template = "What does {thing} smell like?"

chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

# output = chain.run({"thing":"space"},callbacks=[CallingItBack()])
# print(output)

model = ChatOpenAI(streaming=True,temperature=0,verbose=0)
prompt = ChatPromptTemplate.from_template("Answer a question with a strict process and deep analysis {question}")

chain = prompt | model

response = chain.invoke({"question":"who is the walrus?"})
# output = response.content
# print(output)

evaluator = load_evaluator(
    "criteria",
    criteria="relevance",
    llm = ChatOpenAI()
)

eval_result = evaluator.evaluate_strings(prediction="The captial of new york is albany",
                                         input="What is 26+43?")

# print(eval_result)

custom_criteria = {
    "Simplicity":"Does the language use brivity?",
    "bias":"Does the language stay free of human bias?",
    "clarity":"Is the writing easy to understand",
    "truthfulness":"Is the writing honest and factual?"
}

evaluator1 = load_evaluator("criteria",criteria=custom_criteria,llm=ChatOpenAI())

eval_result1 = evaluator1.evaluate_strings(
    input="What is the best Intalian restaurant in the New york City?",
    prediction="That is a subjective statement and I cannot answer that"
)

# print(eval_result1)

loader = PyPDFLoader("attention is all you need.pdf")
data = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=['.'],
)

doc = splitter.split_documents(data)
embeddings = OpenAIEmbeddings()

docstorage = Chroma.from_documents(doc,embeddings)
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docstorage.as_retriever(),
    input_key="question"
)

question_set = [
    {
        "question":"What is the primary presented in the document?",
        "answer":"The Transformer."
    },
    {
        "question":"According to the document, is the Transformer faster or slower than architectures"
                   "based on recurrent or convolutions layers?",
        "answer":"The Transformer is faster."
    },
    {
        "question":"Who is the primary author of the document?",
        "answer":"Ashish vaswani"
    }
]

predictions = qa.apply(question_set)

eval_chain = QAEvalChain.from_llm(llm)

results = eval_chain.evaluate(
    question_set,
    predictions,
    question_key="question",
    prediction_key="result",
    answer_key='answer'
)

print(results)

