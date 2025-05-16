from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

loader = DirectoryLoader(
    path='books',
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)

docs= loader.load()

print(docs[325].page_content)

prompt= PromptTemplate(
    template='Answer the questions \n {question} from following document \n{text}',
    input_variables=['text','question']
)
parser = StrOutputParser()

model = ChatOpenAI()
""" but this takes a lots of time
to solve this problem we use lazy loading
it uses generator of documents , one at time"""

docs = loader.lazy_load()

""" web base loader for static websites"""
#loader1 = WebBaseLoader(url)

chain = prompt|model|parser

chain.invoke({'question':'what is the name ','text':docs[0].page_content})

