# import os
# from apikey import apikey
# import asyncio
# import streamlit as st
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory

# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())


# os.environ['GOOGLE_API_KEY'] = apikey

# st.title('Chat with Your Documents')

# loader = TextLoader('./constitution.txt')
# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks = text_splitter.split_documents(documents)


# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# vector_store = Chroma.from_documents(chunks, embeddings)

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
# retriever = vector_store.as_retriever()

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=retriever,
#      memory= memory,
# )

# user_question = st.text_input("Ask a question about the document:")


# if user_question:
#     response = qa_chain.run(user_question)

#     if 'history' not in  st.session_state:
#         st.session_state.history = []
#         st.session_state.history.append({"role": "user", "content": user_question})
#     st.write("Answer:", response)


# for prompt in st.session_state.history:
#         st.write("Question:", prompt[0])
#         st.write("Answer:", prompt[1])


import os
from apikey import apikey
import asyncio
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

os.environ['GOOGLE_API_KEY'] = apikey

st.title('Chat with Your Documents')

loader = TextLoader('./constitution.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma.from_documents(chunks, embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
retriever = vector_store.as_retriever()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
)

if 'history' not in st.session_state:
    st.session_state.history = []

user_question = st.text_input("Ask a question about the document:")

if user_question:
    response = qa_chain.run(user_question)

    st.session_state.history.append({"role": "user", "content": user_question})
    st.session_state.history.append({"role": "assistant", "content": response})

    st.write("Answer:", response)

# for entry in st.session_state.history:
#     role = entry['role'].capitalize()
#     content = entry['content']
#     st.write(f"{role}: {content}")
