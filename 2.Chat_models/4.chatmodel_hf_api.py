from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

# Create a pipeline for text generation without device_map
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # Removed device_map="auto"
)

# Configure the pipeline parameters
llm = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# For TinyLlama, format the prompt correctly
prompt = """<|system|>
You are a helpful assistant.
<|user|>
What is the capital of France?
<|assistant|>"""

# Generate response
result = llm.invoke(prompt)
print(result)