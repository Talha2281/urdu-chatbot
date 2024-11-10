import os
import numpy as np
import requests
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import streamlit as st
import faiss
from bs4 import BeautifulSoup
import re

# Function to extract text from a webpage
def extract_text_from_html(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = "\n".join([p.get_text() for p in paragraphs])
            return text
        else:
            return ""
    except Exception as e:
        return f"Error extracting content: {e}"

# Recursive URL loader function
def recursive_loader(url, max_depth=2):
    visited = set()
    documents = []
    
    def crawl(url, depth=0):
        if depth > max_depth or url in visited:
            return
        visited.add(url)
        
        text = extract_text_from_html(url)
        if text:
            documents.append(text)
        
        # Find links to continue crawling
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        
        for link in links:
            # Ensure links are valid and from the Rekhta website
            if link.startswith('http') and 'rekhta.org' in link:
                crawl(link, depth + 1)
    
    crawl(url)
    return documents

# Extracting the data from the website
documents = recursive_loader('https://www.rekhta.org/')

# Create embeddings using Hugging Face's model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = embedding_model.embed_documents(documents)  # Use embed_documents instead of embed_text

# Create a vector store using FAISS
index = faiss.IndexFlatL2(len(doc_embeddings[0]))  # Initialize FAISS index with the embedding size
index.add(np.array(doc_embeddings))  # Add embeddings to the index

# Create a FAISS vector store
faiss_index = FAISS(embedding_model, index)  # Create FAISS vector store with embeddings

# Set up a Chat model
chat_model = ChatOpenAI(temperature=0.5)

# Define a prompt template for the RAG app
prompt_template = "You are an AI chatbot trained on content from Rekhta. Answer the following question: {question}"
prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

# Set up the Streamlit interface
st.title('Rekhta Literature Chatbot')
user_input = st.text_input("Ask a question:")

if user_input:
    # Perform the retrieval from the vector store
    results = faiss_index.similarity_search(user_input, k=3)
    context = "\n".join([result.page_content for result in results])

    # Generate the response based on the context
    agent = initialize_agent([Tool(name="Rekhta Knowledge Base", func=context, description="Retrieve data from Rekhta.")], chat_model, verbose=True)
    response = agent.run(user_input)
    
    st.write(response)
