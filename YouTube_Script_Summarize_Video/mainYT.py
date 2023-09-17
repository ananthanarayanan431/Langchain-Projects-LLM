
from constant import openai_key
import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain,SimpleSequentialChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains.conversation.memory import ConversationKGMemory
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.utilities import WikipediaAPIWrapper
import os

os.environ['OPENAI_API_KEY'] = openai_key

st.title("YouTube Scipt Generator With Langchain")

prompt = st.text_input("Enter your Prompt")

# llm = OpenAI(model_name='text-davinci-003',
#              temperature=0.9,
#              max_token=256)

llm = OpenAI(temperature=0.9)

title_mem=ConversationBufferMemory(input_key="Topic",memory_key="title_hist")

prompt1 = PromptTemplate(
    input_variables=['Topic','wikipedia_research'],
    template="Write a YouTube Video tittle on {Topic} while levarging the Wikipedia research {wikipedia_research}"
)

chain1 = LLMChain(llm=llm,prompt=prompt1,verbose=True,output_key="title",memory=title_mem)

scipt_mem =ConversationBufferMemory(input_key="title",memory_key="script_hist")

prompt2 = PromptTemplate(
    input_variables=['title'],
    template="Write a youtube video scipt based on the title {title}"
)

chain2 = LLMChain(llm=llm,prompt=prompt2,verbose=True,output_key="script",memory=scipt_mem)

# parent = SequentialChain(chains=[chain1,chain2],
#                          input_variables=['Topic'],
#                          output_variables=['title','script'],
#                          verbose=True)

wiki = WikipediaAPIWrapper()

if prompt:
    # response = parent({'Topic':prompt})
    # st.write(response['title'])
    # st.write(response['script'])

    title = chain1.run(prompt)
    wiki_research = wiki.run(prompt)
    scipt = chain2.run(title=title,wikipedia_research=wiki_research)
    st.write(title)
    st.write(scipt)

    with st.expander("Title Memory"):
        st.info(title_mem.buffer)

    with st.expander("Scipt Memory"):
        st.info(scipt_mem.buffer)
    with st.expander("Wikipedia information"):
        st.info(wiki_research)