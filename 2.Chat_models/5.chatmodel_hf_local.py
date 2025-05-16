from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

# os.environ['HF_HOME'] = 'D/huggingface_cache' 
# # if you dont have space in c drive and change the default location# Set the cache directory for Hugging Face models

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=50,
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of France?")
print(result.content)

""" this type of model is very big and takes lots of time
to run downloading models in not a good practice"""