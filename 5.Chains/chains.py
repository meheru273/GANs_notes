from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template = 'Generate 5 interesting facts about {topic}',
    input_variables = ['topic']
)

model =  ChatOpenAI()

parser = StrOutputParser()

#simple chain 
chain = prompt | model | parser
result = chain.invoke({"topic": "Python programming"})
#for visualization of chain 
chain.get_graph().print_ascii()

#sequential chain 
prompt1 = PromptTemplate(
    template = 'Generate detailed report on  {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate short summary on following text :  {text}',
    input_variables = ['text']
)

seq_chain = prompt1 | model | parser| prompt2 | model | parser
result2 = seq_chain.invoke({"topic": "Python programming"})