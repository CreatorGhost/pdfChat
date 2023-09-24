import os
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
import redis
from langchain.vectorstores import Redis
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

load_dotenv() 

openai.api_key = os.getenv("OPENAI_API_KEY")

# response = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {
#       "role": "system",
#       "content": "Hey how will ai chage the world say in in one sentence "
#     },
#     {
#       "role": "assistant",
#       "content": "\n"
#     }
#   ],
#   temperature=1,
#   max_tokens=256,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )

# print(response["choices"][0]["message"]["content"])


chat_model = ChatOpenAI(model="gpt-4")

# ans = chat_model.predict("how was our univers created")


loader = PyPDFLoader("covid.pdf")
pages = loader.load_and_split()

redis_instace = redis.Redis(
    host ="127.0.0.1",
    port= "6379",
)


document_store = Redis.from_documents(pages, OpenAIEmbeddings(),index_name="covid",redis_url="redis://127.0.0.1:6379")

retriever = document_store.as_retriever()

qa = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=retriever)

ans = qa.run("Summarise the pdf file")

print(ans)

# search_res = document_store.similarity_search("When and where was covid detected ")

