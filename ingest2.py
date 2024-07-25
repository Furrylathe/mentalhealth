import os
import pandas as pd
import json
import logging
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()

def get_csv_text(file_path):
    df = pd.read_csv(file_path)
    return "\n".join(df.astype(str).apply(lambda x: ' '.join(x), axis=1))

def get_json_text(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return "\n".join([f"{item['tag']} {item['patterns']} {item['responses']}" for item in data['intents']])

def get_nlp_mental_health_text(file_path):
    df = pd.read_csv(file_path)
    return "\n".join(df.astype(str).apply(lambda x: f"Context: {x['Context']} Response: {x['Response']}", axis=1))

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Initialize FAISS
    dimension = 768  # Set the dimensionality of the embeddings
    index = faiss.IndexFlatL2(dimension)
    
    # Prepare to collect all embeddings and texts
    all_embeddings = []
    all_texts = []
    
    for i in range(0, len(text_chunks), 1000):
        chunk_batch = text_chunks[i:i + 1000]
        try:
            logging.info(f"Processing batch {i // 1000 + 1} with {len(chunk_batch)} chunks.")
            embeddings_batch = embeddings.embed_documents(chunk_batch)
            all_embeddings.extend(embeddings_batch)
            all_texts.extend(chunk_batch)
            logging.info(f"Processed batch {i // 1000 + 1} with {len(chunk_batch)} chunks.")
        except Exception as e:
            logging.error(f"Failed to process batch {i // 1000 + 1}: {str(e)}")
            break
    
    # Convert embeddings to numpy array
    all_embeddings_np = np.array(all_embeddings).astype('float32')
    
    # Add embeddings to the FAISS index
    index.add(all_embeddings_np)
    
    # Save the FAISS index
    faiss.write_index(index, "faiss.index")
    
    # Save texts
    with open("texts.txt", "w") as f:
        for text in all_texts:
            f.write(f"{text}\n")

def test_embeddings():
    text = "This is a test document."
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        embedding = embeddings.embed_documents([text])
        logging.info(f"Embedding successful: {embedding}")
    except Exception as e:
        logging.error(f"Failed to create embedding: {str(e)}")

def ingest_data():
    logging.info("Starting data ingestion process.")

    csv_text = get_csv_text("dataset/emotion_sentimen_dataset.csv")
    logging.info("Finished reading CSV file.")
    
    json_text = get_json_text("dataset/intents.json")
    logging.info("Finished reading JSON file.")
    
    nlp_mental_health_text = get_nlp_mental_health_text("dataset/nlp_mental_health.csv")
    logging.info("Finished reading NLP Mental Health CSV file.")

    all_text = "\n".join([csv_text, json_text, nlp_mental_health_text])
    text_chunks = get_text_chunks(all_text)
    logging.info(f"Created {len(text_chunks)} chunks.")
    
    logging.info("Creating embeddings for text chunks.")
    create_vector_store(text_chunks)
    logging.info("Finished creating embeddings and saving vector store.")

if __name__ == "__main__":
    test_embeddings()
    ingest_data()
