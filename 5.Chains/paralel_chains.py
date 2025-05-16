from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
load_dotenv()

model1 = ChatOpenAI()

model2 = ChatAnthropic()

prompt1 = PromptTemplate(
    template = 'Generate short and simple summary of the following text {topic}',
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template = 'Generate 5 short quize from following text  {topic}',
    input_variables = ['topic']
)
prompt3 = PromptTemplate(
    template = 'Merge the summray and the quize into single document \n notes-> {notes} \n quize-> {quize}',
    input_variables = ['notes', 'quize']
)
parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quize": prompt2 | model2 | parser,
    }
)
merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain
result = chain.invoke({"topic": "Python programming"}) 
 