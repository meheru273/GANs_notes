from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal
load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

class feedback(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback {feedback} \n {format_instruction}',
    input_variables = ['feedback'],
    partial_variables= {"format_instruction": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

#result = classifier_chain.invoke({"feedback": "The product is great!"}).sentiment
prompt2 = PromptTemplate(
    template = 'Generate a response for the following positive feedback {feedback}',
    input_variables = ['feedback']
)
prompt2 = PromptTemplate(
    template = 'Generate a response for the following negative feedback {feedback}',
    input_variables = ['feedback']
)
branch_chain = RunnableBranch(
    (lambda x: x['sentiment'== "positive",prompt2 | model | parser] ),
    (lambda x: x['sentiment'== "negative",prompt2 | model | parser] ),
    RunnableLambda(lambda x: "could not find sentiment")
    
)

chain = classifier_chain | branch_chain
result = chain.invoke({"feedback": "The product is great!"})