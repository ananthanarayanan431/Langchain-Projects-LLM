

from typing import Union, List
from dotenv import load_dotenv
from langchain_core.agents import AgentAction, AgentFinish

load_dotenv()
import os

os.environ["OPENAI_API_KEY"] = ""

from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from callbacks import AgentCallbackHandler

@tool
def get_text_length(text: str) -> int:
    """Return the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    return len(text)

@tool
def write_haiku(topic:str)->str:
    """Writes a haiku about a given topic."""
    print(f"write haiku enter with {topic=}")
    return ChatOpenAI().predict(text=f"Write a haiku about {topic}")

def find_tool_by_name(tools: List[tool], tool_name: str) -> tool:
    for t in tools:
        if t.name == tool_name:
            return t
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    print("React Langchain")
    tools = [
        get_text_length,
        write_haiku,
    ]

    template = """
    Answer the following questions as best you can. You have access to the following tools:
    
    {tools}
    
    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        stop=["\nObservation"],
        callbacks=[AgentCallbackHandler()],
    )

    intermediate_steps = []


    agent = (
        {"input": lambda x: x["input"],
         'agent_scratchpad':lambda x:format_log_to_str(x['agent_scratchpad'])
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""

    while not isinstance(agent_step,AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {"input": "What is the length of 'Anantha' in characters?",
             'agent_scratchpad':intermediate_steps,
            }
        )
        print(agent_step)
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")
            intermediate_steps.append((agent_step,str(observation)))

    if isinstance(agent_step,AgentFinish):
        print(agent_step.return_values)
