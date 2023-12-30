
import os
from constant import anantha_api

os.environ['OPENAI_API_KEY']=anantha_api


from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

from langchain import FewShotPromptTemplate

llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0
)

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}?
"""

prompt = PromptTemplate(
    input_variables = ['theme','year'],
    template = template
)

input_data = {'theme':'Titanic Travel', 'year':'2023'}

chain = LLMChain(
    llm=llm,
    prompt=prompt
)

response = chain.run(input_data)

print("Theme : Titanic Travel")
print("Year : 2023")

print("AI-generated song title:", response)

examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"},
]

example_formatter_template = """
Color: {color}
Emotion: {emotion}\n
"""

example_prompt = PromptTemplate(
    input_variables=["color", "emotion"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Here are some examples of colors and the emotions associated with them:\n\n",
    suffix="\n\nNow, given a new color, identify the emotion associated with it:\n\nColor: {input}\nEmotion:",
    input_variables=["input"],
    example_separator="\n",
)

formatted_prompt = few_shot_prompt.format(input="purple")

chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=formatted_prompt,
        input_variables=[]
    ) 
)

response = chain.run({})

print("Color: Purple")
print("Emotion: ",response)