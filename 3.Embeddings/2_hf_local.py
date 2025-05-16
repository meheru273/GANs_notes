from langchain.embeddings import HugginFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = HugginFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

text = "The capital of France is Paris."
result = embedding.embed_query(text)
print(str(result))

#document embedding
documents = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome."
]   

results = embedding.embed_documents(documents)
print(str(results)) 