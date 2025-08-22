import os
from apikey import apikey

# import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import load_tools, initialize_agent, AgentType
os.environ['GOOGLE_API_KEY'] = apikey


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
tools = load_tools(['wikipedia'], llm=llm)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

prompt = input("wikipedia research task:  ")

result =agent.run(prompt)
print(result)
