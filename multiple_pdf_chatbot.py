
import streamlit as st
from PyPDF2 import PdfReader
from torch import cuda
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub, HuggingFacePipeline
import openai



##!pip install streamlit PdfReader load_dotenv torch PyPDF2 -U langchain-community openai faiss-cpu

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=60,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

import os
# Set the environment variable for OPENAI_API_KEY
api_key = ''
os.environ['OPENAI_API_KEY'] = api_key

## uploading the pdf files
file_paths_input = input("Enter the path to your PDF files, separated by commas: ")
file_paths = file_paths_input.split(',')
file_paths = [path.strip() for path in file_paths]

def main():

    # Process PDF files
    raw_text = get_pdf_text(file_paths)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation = get_conversation_chain(vectorstore)

    while True:
        user_question = input("Ask a question about your documents (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            print("Exiting the conversation.")
            break
        # Get answer using the invoke method
        response = conversation.invoke({'question': user_question})
        print("Raw Response:", response)

if __name__ == '__main__':
    main()