""" if the text has more than 500 words then it will be summarized"""
from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnableSequence,RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

prompt1 = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"],
)
prompt2 = PromptTemplate(
    template="Summarize the text : \n {topic}",
    input_variables=["topic"],
)

model = ChatOpenAI()
parser = StrOutputParser()

gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)
final_chain = RunnableSequence(gen_chain,branch_chain)

final_chain.invoke({'topic' : 'python programming'})