import streamlit as st
from langchain import HuggingFaceHub
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
import pypdf
from langchain_community.document_loaders import PyPDFLoader

file_path = "NRUP_content.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

# Create the LLM
llm=HuggingFaceHub(repo_id="deepseek-ai/DeepSeek-R1", model_kwargs={"temperature":0.01})

# Create the Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

retriever = vectorstore.as_retriever()
