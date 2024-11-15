import os
import requests
import numpy as np
from langchain.agents import initialize_agent, Tool
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import streamlit as st
import faiss
from bs4 import BeautifulSoup
import re

# GIMNI API key and URL for testing purposes
gimni_api_key = ""  # Directly embedding for testing
gimni_api_url = "https://api.gimni.com/v1/ask"  # Replace with the actual GIMNI API URL if different

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

# Create a FAISS vector store with docstore and index_to_docstore_id
docstore = {i: doc for i, doc in enumerate(documents)}  # Map each document to an ID
index_to_docstore_id = {i: str(i) for i in range(len(docstore))}  # Map FAISS index to docstore ID

faiss_index = FAISS(embedding_model, index, docstore, index_to_docstore_id)  # Initialize FAISS with necessary parameters

# Function to query the GIMNI API
def query_gimni_api(question, context):
    headers = {
        "Authorization": f"Bearer {gimni_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "question": question,
        "context": context
    }
    response = requests.post(gimni_api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("answer", "Sorry, I couldn't find an answer.")
    else:
        return "Error querying GIMNI API."

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

    # Generate the response based on the context using GIMNI API
    response = query_gimni_api(user_input, context)
    
    st.write(response)




      
        








  


   

          

