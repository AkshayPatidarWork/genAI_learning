import os
from apikey import apikey

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
os.environ['GOOGLE_API_KEY'] = apikey


st.title('Medium Article Generator')
topic = st.text_input("Enter a topic for your article:")



if topic:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    response = llm.invoke(topic)

    st.subheader("Generated Article:")
    st.write(response.content)
