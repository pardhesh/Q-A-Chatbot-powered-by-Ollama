from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import  StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

os.environ["LANGCHAIN_API"]=os.getenv("LANGCHAIN_API")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q/A Chatbot through Ollama"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistnt that answers the questions"),
        ("user","Question{query}")
    ]
)

def generate_response(query,llm_name,temperature,max_tokens):
    model = Ollama(model=llm_name)
    output = StrOutputParser()
    chain = prompt|model|output
    answer = chain.invoke({'query':query})
    return answer

 
st.title("Simple Q/A Chatbot through Ollama")

st.sidebar.title("Parameters")
llm_name = st.sidebar.selectbox("Select the model", ["gemma3:1b","Mistral2:7b"])

temperature = st.sidebar.slider("Temperature",min_value=0,max_value=100,value =4)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=200,value = 150)

st.write("Ask the question")
user_input = st.text_input("You ")

if user_input:
    response = generate_response(user_input,llm_name,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please enter the question.")

