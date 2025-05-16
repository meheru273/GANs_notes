from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain_community.prompts import PromptTemplate

load_dotenv()

st.header("research tool")

paper_input = st.selectbox("select paper name", ["paper1", "paper2", "paper3"])
style_input = st.selectbox("select style",      ["style1", "style2", "style3"])
length_input = st.selectbox("select length",    ["short", "medium", "long"])

# Correct PromptTemplate
template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template="""
You are a research assistant.
You are given a paper name, style and length.
Your task is to write a summary of the paper in the given style and length.

Paper name: {paper_input}
Style:      {style_input}
Length:     {length_input}
""",  # <-- comma here is essential
)

template.save('template.json')
""" we can save templates as json files
and use them later by loading the files"""

# Fill the template
prompt = template.invoke({
    "paper_input":  paper_input,
    "style_input":  style_input,
    "length_input": length_input,
})

# Instantiate your LLM
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

if st.button("Submit"):
    result = model.invoke(prompt)
    st.write(result.content)

    """if st.button("submit"):
        chain = template|model
        result = chain.invoke({
            "paper_input":  paper_input,
            "style_input":  style_input,
            "length_input": length_input,
        })
        st.write(result.content)"""
    """why use PrompTemplate?
    1. Default Validation 
    2. Reusable 
    3. LangChain Ecosystem Integration
    4. Customization
    """