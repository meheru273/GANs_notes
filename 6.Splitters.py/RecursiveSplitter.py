from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from langchain_community.document_loaders import PyPDFLoader

splitter = RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size = 300,
    chunk_overlap =0,
)

chunk = splitter.split_text()