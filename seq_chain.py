import os
from apikey import apikey

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,  SequentialChain

os.environ['GOOGLE_API_KEY'] = apikey


st.title('Medium Article Generator')
topic = st.text_input("Enter a topic for your article:")
language = st.text_input('Enter the language for the article (e.g., English, Hindi):', 'English')

title_template = PromptTemplate(
    input_variables=["topic", "language"],
    template="give me a  Medium article title on  {topic} in {language} language."
)

article_template = PromptTemplate(
    input_variables=["title", "language"],
    template="Write a Medium article on {title} in {language} language."
)

if topic:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    title_chain = LLMChain(llm=llm, prompt=title_template, output_key="title")

    article_chain = LLMChain(llm=llm, prompt=article_template, output_key="article")

    seq_chain = SequentialChain(chains=[title_chain, article_chain],     input_variables=["topic", "language"],
        output_variables=["title", "article"], verbose=True)
    
    response = seq_chain.invoke({"topic": topic, "language": language})

    st.subheader("Generated Title:")
    st.write(response["title"])
    
    st.subheader("Generated Article:")
    st.write(response["article"])
