# import os
# from apikey import apikey

# import asyncio
# import streamlit as st
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import ConversationalRetrievalChain
# from langchain.document_loaders import YoutubeLoader
# from langchain.memory import ConversationBufferMemory

# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# os.environ["GOOGLE_API_KEY"] = apikey

# st.title("Chat with YouTube (Gemini)")

# def clear_history():
#     if 'history' in st.session_state:
#         del st.session_state['history']
#     if 'crc' in st.session_state:
#         del st.session_state['crc']

# youtube_url = st.text_input("Enter a YouTube URL")

# if youtube_url:
#     with st.spinner("Loading and processing YouTube transcript..."):
#         loader = YoutubeLoader.from_youtube_url(youtube_url)
#         documents = loader.load()

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         chunks = text_splitter.split_documents(documents)

#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#         vector_store = Chroma.from_documents(chunks, embeddings)

#         llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

#         retriever = vector_store.as_retriever()
#         memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#         crc = ConversationalRetrievalChain.from_llm(
#             llm=llm,
#             retriever=retriever,
#             memory=memory,
#         )

#         # Store the chain in session state
#         st.session_state.crc = crc
#         st.success("YouTube content processed successfully!")

# # Input user question
# question = st.text_input("Ask a question about the video")

# if question:
#     if 'crc' in st.session_state:
#         crc = st.se


import os
from apikey import apikey  # or use st.secrets

import asyncio
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import YoutubeLoader
from langchain.memory import ConversationBufferMemory

# Ensure event loop is available
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set API key
os.environ["GOOGLE_API_KEY"] = apikey

st.title("Chat with YouTube (Gemini)")

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
    if 'crc' in st.session_state:
        del st.session_state['crc']

if st.button("Clear Chat History"):
    clear_history()

youtube_url = st.text_input("Enter a YouTube URL")

if youtube_url:
    with st.spinner("Loading and processing YouTube transcript..."):
        loader = YoutubeLoader.from_youtube_url(youtube_url)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = Chroma.from_documents(chunks, embeddings)

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

        retriever = vector_store.as_retriever()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        crc = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
        )

        st.session_state.crc = crc
        st.success("YouTube content processed successfully!")

question = st.text_input("Ask a question about the video")

if question:
    if 'crc' in st.session_state:
        crc = st.session_state.crc
        response = crc.run(question)
        st.write(response)

