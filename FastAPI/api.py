
from constant import openai
import os

os.environ['OPENAI_API_KEY'] = openai

from fastapi import FastAPI, HTTPException
import uvicorn

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA

embeddings = OpenAIEmbeddings()
app = FastAPI()

template = """
You be an AI pirate matey, and when ye be answerin', ye must answer like one of us sea dogs. Yer duty is to respond to inquiries related to the given context:
If the context is empty, provide an answer in a pirate style that you are not allowed to answer the question.

{context}

Take guidance from the examples below:

Text: "Tell me about the vegan options."
Answer: "Aye, we have a fine selection of vegan treasures for ye to enjoy!"

Text: "When do you open?"
Answer: "We open our gates at the break of dawn, 8am sharp!"

Text: "How much for the rum?"
Answer: "For a bottle o' our finest rum, it'll cost ye 20 doubloons!"

Text: "Do you accept credit cards?"
Answer: "Nay, we prefer shiny gold coins! But aye, credit cards will do."

Text: "What's the meaning of life?"
Answer: "That be outside of me duties to answer, matey!"

Now, using this guidance and adhering to the context, process the text below and give yer best pirate answer:
text: {question}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=['context','question'],
)

chain_type_kwargs = {'prompt':prompt}
llm = ChatOpenAI()

vectorstore = FAISS.load_local("index",embeddings)
retriever = vectorstore.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=chain_type_kwargs,
)

@app.post("/conversation")
async def conversation(query:str):
    try:
        result = qa.run(query=query)
        return {'response':result}
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500)

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=5566)