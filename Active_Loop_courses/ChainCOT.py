
import os
from constant import anantha_api

os.environ['OPENAI_API_KEY']=anantha_api

from langchain import FewShotPromptTemplate, PromptTemplate, LLMChain
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003",temperature=0)

examples = [
    {
        "query": "What's the secret to happiness?",
        "answer": "Finding balance in life and learning to enjoy the small moments."
    }, {
        "query": "How can I become more productive?",
        "answer": "Try prioritizing tasks, setting goals, and maintaining a healthy work-life balance."
    }
]

example_template = """
user : {query}
AI : {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query","answer"],
    template = example_template
)

prefix = """The following are excerpts from conversations with an AI
life coach. The assistant provides insightful and practical advice to the users' questions. Here are some
examples: 
"""

suffix = """
User: {query}
AI: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)

user_query = "What are some tips for improving communication skills?"

response = chain.run({"query": user_query})

print("User Query:", user_query)
print("AI Response:", response)

