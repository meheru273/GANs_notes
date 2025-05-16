from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type = 'standard_deviation',
    breakpoint_threshold_amount=1
)

sample = """Structured data is data that is organized in
a predefined format—think tables, rows, columns, CSV files, or SQL databases.
It’s the kind of data you can easily plug into a spreadsheet 
or a machine learning model.
Structured Data Generation refers to techniques used to create new,
synthetic structured data that mimics real data. 
"""
chunk= text_splitter.create_documents([sample])