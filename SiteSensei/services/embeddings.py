import pickle
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def manage_embeddings(docs):
    embeddings = OpenAIEmbeddings()
    file_path = "faiss_store_openai.pkl"
    if not os.path.exists(file_path):
        vectorStore_openAI = FAISS.from_documents(docs, embeddings)
        with open(file_path, "wb") as f:
            pickle.dump(vectorStore_openAI, f)
    with open(file_path, "rb") as f:
        return pickle.load(f)
