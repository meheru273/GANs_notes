"""This code tries to find how close the embedding 
of a given sentence is to other vectors in documents"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# Use a free Hugging Face embedding model instead of OpenAI
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming industries worldwide.",
    "Python is a versatile programming language loved by developers.",
    "Climate change poses significant challenges to our global ecosystem.",
    "Space exploration drives innovation and expands our understanding of the universe.",
    "Renewable energy sources like solar and wind are key to reducing carbon emissions.",
    "Blockchain technology offers secure, transparent ways to record transactions.",
    "Advancements in biotechnology are revolutionizing healthcare and medicine.",
    "Virtual reality is creating immersive experiences in gaming and education.",
    "Data privacy concerns are prompting new regulations around the globe.",
    "Electric vehicles are becoming more affordable and environmentally friendly.",
    "Quantum computing promises to solve problems beyond the reach of classical computers.",
    "5G networks enable faster connectivity and support the Internet of Things.",
    "Machine learning algorithms can detect patterns in vast datasets with high accuracy.",
    "Urban agriculture initiatives are helping cities become more sustainable.",
    "Cybersecurity threats are evolving, requiring robust defense strategies.",
    "Augmented reality overlays digital content onto the physical world.",
    "Nanotechnology is being used to develop materials with novel properties.",
    "Autonomous drones are transforming logistics and aerial surveying.",
    "Renewable building materials reduce environmental impact in construction."
]

query = "what is Artificial intelligence?"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]

# Print results sorted by similarity score (highest to lowest)
index,scores = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)

print(query)
print(documents[index])
print(scores)