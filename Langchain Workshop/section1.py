

from constant import huggingface,openai
import os

from langchain_community.llms import HuggingFaceHub
from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct",huggingfacehub_api_token=huggingface)
question = "Can you still have fun?"

llm1 = OpenAI(openai_api_key=openai)


# output = llm.invoke(question)
# print(output)

# output = llm1.invoke(question)
# print(output)


template = "you're an Artifical Intelligence assistant, answer the question. {question}"
prompt = PromptTemplate(template=template,input_variables=['question'])
llm_chain = LLMChain(prompt=prompt,llm=llm)

# question="What is langchain?"
# print(llm_chain.invoke(question))


llm2 = ChatOpenAI(temperature=0,openai_api_key=openai)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

full_prompt = template.format_messages(
    user_input="What is the sound of one hand clapping?",
)
# print(llm2(full_prompt))
# for message in full_prompt:
#     print(message.__repr__())

history = ChatMessageHistory()
history.add_ai_message("Hi! Ask me anything about langchain")
history.add_user_message("Describe a metaphor for learning langchain in one sentence?")
history.add_user_message("Summarize the precedding sentences in fewer words")
# print(history.messages)

memory = ConversationBufferMemory(size=4)
buffer_chain = ConversationChain(llm=llm2,memory=memory,verbose=True)

# buffer_chain.predict(input="Describe a language model in one sentence")
# buffer_chain.predict(input="Describe it again using less words")
# buffer_chain.predict(input="Describe it again fewer words but at least one word")
# buffer_chain.predict(input="What did I first ask you? I forgot")

memory1 = ConversationBufferMemory(
    llm=llm1,
)

summary_chain = ConversationChain(llm=llm2,memory=memory1,verbose=True)
summary_chain.predict(input="Please summarize the future in 2 sentences")
summary_chain.predict(input="Why?")
summary_chain.predict(input="What will I need to shape this?")