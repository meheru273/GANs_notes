from langchain_openai import OpenAIEmbeddings
from langchain.vectorestores import Chroma
from langchain.schema import Document

# Document 1: an article excerpt
doc1 = Document(
    page_content="Artificial intelligence (AI) is transforming industries by enabling machines to learn from data and perform tasks that normally require human intelligence.",
    metadata={
        "source": "Tech Journal",
        "author": "Alice Smith",
        "published": "2025-01-15",
        "topic": "AI Overview"
    }
)

# Document 2: a blog post summary
doc2 = Document(
    page_content="In this post, we explore how to build and deploy a simple Flask API for image classification, including Dockerization and cloud deployment tips.",
    metadata={
        "source": "DevBlog",
        "author": "Bob Chen",
        "published": "2024-11-02",
        "topic": "Flask API"
    }
)

# Document 3: meeting notes
doc3 = Document(
    page_content="Meeting Notes:\n- Reviewed project milestones for the Braille-to-text OCR system.\n- Identified memory bottlenecks when handling >130 segments per image.\n- Agreed to optimize segment pipeline and deploy via a Flask microservice.",
    metadata={
        "source": "Team Drive",
        "meeting_date": "2025-04-20",
        "participants": ["Carol", "Dave", "Eve"],
        "project": "Braille OCR"
    }
)

# Collect into a list
docs = [doc1, doc2, doc3]

vector_store = Chroma(
    embedding_function = OpenAIEmbeddings(),
    persist_directory = 'chroma_Db',
    collection_name = 'sample'
)

vector_store.add_documents(docs)

#view documents

vector_store.get(include=['embeddings','documents','metadata'])

#search_documents

vector_store.similarity_search(
    query = '',
    k=1 , # how many similar objects we want to see
    
)

#similarity scores

vector_store.similarity_search_with_score(
    query = 'who among these are a bowler',
    k=2
)

#update documents 
updated_doc = Document(
    page_content = "AI and AI Agents",
    metadata = {
        "source": "Tech Journal",
        "author": "Alice Smith",
        "published": "2025-01-15",
        "topic": "AI Overview"
    }
)

vector_store.update_document(document_id='',document=updated_doc)

#delete documents

vector_store.delete(ids=[''])