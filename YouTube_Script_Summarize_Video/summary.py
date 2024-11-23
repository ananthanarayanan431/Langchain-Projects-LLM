
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

loader = YoutubeLoader.from_youtube_url(
    youtube_url="https://youtu.be/J4Hd5wudIrk?si=3lEfjwZkxTRkwiHb",
    add_video_info=False,
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=[""," ",'',' '],
    length_function=len,
)
splits = text_splitter.split_documents(docs)
template = """
Write a concise summary of the following 

Give the summary in points
{context}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = create_stuff_documents_chain(llm, prompt)
ans = chain.invoke({'context':docs})
print(ans)
