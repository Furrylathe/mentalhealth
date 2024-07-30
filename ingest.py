
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle
import pandas as pd
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_csv(csv_files):
    text_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if 'text' in df.columns:
            text_data.extend(df['text'].tolist())
        elif 'content' in df.columns:
            text_data.extend(df['content'].tolist())
    return text_data

def extract_text_from_json(json_files):
    text_data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            for intent in data['intents']:
                text_data.extend(intent['patterns'])
    return text_data

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("Faiss")

def main():
    cache_file = "cached_data.pkl"

    # Load from cache if available (avoids reprocessing PDFs)
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            text_chunks = pickle.load(f)
            print("Loaded data from cache.")
    else:
        # Process PDFs and create vector store if cache is missing
        pdf_files = [os.path.join("dataset", file) for file in os.listdir("dataset") if file.endswith(".pdf")]
        csv_files = [os.path.join("datasets", file) for file in os.listdir("datasets") if file.endswith(".csv")]
        json_files = [os.path.join("datasets", file) for file in os.listdir("datasets") if file.endswith(".json")]

        pdf_text = extract_text_from_pdf(pdf_files) if pdf_files else ""
        csv_text = extract_text_from_csv(csv_files) if csv_files else []
        json_text = extract_text_from_json(json_files) if json_files else []

        all_text = pdf_text + " ".join(csv_text) + " ".join(json_text)
        text_chunks = get_text_chunks(all_text)
        create_vector_store(text_chunks)

        # Save processed data to cache for future use
        with open(cache_file, "wb") as f:
            pickle.dump(text_chunks, f)
            print("Saved data to cache.")

if __name__ == "__main__":
    main()

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs):
  text = ""
  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text

def get_text_chunks(text):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
  chunks = text_splitter.split_text(text)
  return chunks

def create_vector_store(text_chunks):
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
  vector_store.save_local("Faiss")

def main():
  cache_file = "cached_data.pkl"

  # Load from cache if available (avoids reprocessing PDFs)
  if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
      text_chunks = pickle.load(f)
      print("Loaded data from cache.")
  else:
    # Process PDFs and create vector store if cache is missing
    pdf_files = []
    for file in os.listdir("dataset"):
      if file.endswith(".pdf"):
        pdf_files.append(os.path.join("dataset", file))

    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    create_vector_store(text_chunks)

    # Save processed data to cache for future use
    with open(cache_file, "wb") as f:
      pickle.dump(text_chunks, f)
      print("Saved data to cache.")

if __name__ == "__main__":
  main()

