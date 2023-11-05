import os
from third_party.constant import kamalesh_oepnai

os.environ['OPENAI_API_KEY'] = kamalesh_oepnai

from langchain.agents import AgentType,create_csv_agent,initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools import PythonREPLTool,Tool

def main():
    print("Start....")
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    #query = "Generate and save in current working directory of 3 QRcodes that points to https://www.linkedin.com/in/rananthanarayananofficial/ . you have qrcode package installed"
    # python_agent_executor.run("Create a python code to run dynamic programming")
    #python_agent_executor.run(query)

    csv_agent_executor = create_csv_agent(
        llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo"),
        path="Balaji Fast Food Sales.csv",
        verbose=True,
        agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    csv_agent_executor.run("Number of columns in CSV file")


    grand_agent = initialize_agent(
        tools=[
            Tool(
                name="PythonAgent",
                func=python_agent_executor.run,
                description="""useful when you need to transform natural language and write from it python and execute the python code,
                                              returning the results of the code execution,
                                            DO NOT SEND PYTHON CODE TO THIS TOOL"""
            ),
            Tool(
                name="CSVAgent",
                func=csv_agent_executor.run,
                description="""useful when you need to answer question over Balaji Fast Food Sales.csv file,
                                             takes an input the entire question and returns the answer after running pandas calculations"""
            ),
        ],
        llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo"),
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
    query = "Generate and save in current working directory of 3 QRcodes that points to https://www.linkedin.com/in/rananthanarayananofficial/ . you have qrcode package installed"
    grand_agent.run(query)

if __name__=='__main__':
    main()