
from openai import OpenAI
import os
from constant import anantha_api

os.environ['OPENAI_API_KEY'] = anantha_api

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role":"system","content":"you're helpful assisstant."},
        {"role":"user","content":"Tell me about India"}
    ]
)

print(completion.choices[0].message.content)