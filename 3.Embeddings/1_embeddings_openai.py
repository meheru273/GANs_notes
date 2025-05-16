from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model="text-embedding-ada-002", dimensions=32)

result = embedding.embed_query("What is the capital of France?")

print(str(result))

#embedding document

documents = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome."
]

results = embedding.embed_documents(documents)
print(str(results))