import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

load_dotenv()


# Initialize a classifier pipeline
classifier = pipeline("text-classification", model="textattack/bert-base-uncased-SST-2")

def is_in_scope(query):
    result = classifier(query)
    label = result[0]['label']
    if label == 'NEGATIVE':
        return False
    return True

# Load data and vector store (assuming these are pre-processed using ingest.py)
def get_data():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("Faiss", embeddings, allow_dangerous_deserialization=True)
    return new_db

def get_conversational_chain():
    prompt_template = """
    You are an experienced psychologist providing mental health care advice based on the provided context. 
    You will respond to the user's queries by leveraging your psychological expertise and the Context Provided.
    Provide empathetic, supportive, and actionable advice based on the user's needs.
    Context: {context}
    Chat History: {chat_history}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.6,
        system_instruction="You are an experienced psychologist providing mental health care advice based on the provided context. You will respond to the user's queries by leveraging your psychological expertise and the Context Provided.")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chat_history, new_db):
    if not is_in_scope(user_question):
        return "I'm sorry, but I can only provide advice on mental health-related topics. For other information, please refer to other sources."

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "chat_history": chat_history, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config("Mental Health Care AI", page_icon=":brain:")
    st.header("Mental Health Care AI :brain:")

    new_db = get_data()

# Load data and vector store (assuming these are pre-processed using ingest.py)
def get_data():
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  new_db = FAISS.load_local("Faiss", embeddings, allow_dangerous_deserialization=True)
  return new_db

def get_conversational_chain():
  prompt_template = """
  You are an experienced psychologist providing mental health care advice based on the provided context. 
  You will respond to the user's queries by leveraging your psychological expertise and the Context Provided.
  Provide empathetic, supportive, and actionable advice based on the user's needs.
  Context: {context}
  Chat History: {chat_history}
  Question: {question}
  Answer:
  """
  model = ChatGoogleGenerativeAI(
      model="gemini-1.5-flash-latest",
      temperature=0.6,
      system_instruction="You are an experienced psychologist providing mental health care advice based on the provided context. You will respond to the user's queries by leveraging your psychological expertise and the Context Provided.")
  prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
  chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
  return chain

def user_input(user_question, chat_history, new_db):
  docs = new_db.similarity_search(user_question)
  chain = get_conversational_chain()
  response = chain.invoke({"input_documents": docs, "chat_history": chat_history, "question": user_question}, return_only_outputs=True)
  return response["output_text"]

def main():
  st.set_page_config("Mental Health Care AI", page_icon=":heart:")
  st.header("Mental Health Care AI :heart:")

  new_db = get_data()


  if "messages" not in st.session_state:
    st.session_state.messages = [
      {"role": "assistant", "content": "Hi I'm your AI Mental Health Advisor"}]

  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.write(message["content"])

  prompt = st.chat_input("Type your question here...")
  if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.write(prompt)


        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    response = user_input(prompt, chat_history, new_db)
                    st.write(response)

                if response is not None:
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)

    if st.session_state.messages[-1]["role"] != "assistant":
      with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
          chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
          response = user_input(prompt, chat_history, new_db)
          st.write(response)

        if response is not None:
          message = {"role": "assistant", "content": response}
          st.session_state.messages.append(message)


if __name__ == "__main__":
  main()
