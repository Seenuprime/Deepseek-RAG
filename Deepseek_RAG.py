from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

st.title("Welcome to DeepSeek integrated with ChatGroq")

api = st.sidebar.text_input("Please enter your Groq API Key here:", type='password')
if api:
    llm = ChatGroq(model='deepseek-r1-distill-qwen-32b', api_key=api)

# # Load the text file and chunk the document
# loader = TextLoader('data.txt')
# docs = loader.load()

# Load the Pdf file and chunk the document
pdf_loader = PyMuPDFLoader('attention_all_youneed.pdf')
docs = pdf_loader.load()

# Split the docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separators='/n')
chunks = text_splitter.split_documents(docs)

# Initialize embedding and store in FAISS vector StopIteration
embedding_model = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(chunks, embedding_model)

prompt_template = PromptTemplate.from_template(
     """

    You are an AI assistant with access to retrieved knowledge. Based on the following context, answer the question,

    make answers look attractive, and more readable,

    if you can't find the answer to that question you just say i can't find the answer in the given context,

    but you can give answer by what you know saying "here is my answer":  

    \n---  

    {context}  

    \n---  

    \nUser Question: {query}

    """ 
)

query = st.text_input("Enter you query: ")

if query and api:
    docs = vector_store.similarity_search(query, k=3)
    retrieved_context = '/n'.join([doc.page_content for doc in docs])    

    prompt = prompt_template.format(context=retrieved_context, query=query)
    
    ai_message = llm.invoke([prompt])
    content = ai_message.content
    actual_content = content.split("</think>")[1]

    if "<think>" in content and "</think>" in content:
        think_part = content.split("</think>")
        think_content = think_part[0].replace("<think>", "")
        print(think_content)
    else:
        think_content = None

    if think_content:
        st.markdown(
            f"""
            <details>
                <summary style="font-size: 14px; font-weight: bold; cursor: pointer; color: #007BFF;">💡 Click to see AI's Thought Process</summary>
                <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.2; color: #f2c14e; padding: 10px; border-radius: 5px; margin-top: 10px;">
                    {think_content}</div></details>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
                details summary {
                    outline: none;
                }
                details:hover summary {
                    color: #0056b3;
                }
                details[open] summary {
                    color: #FF5733;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

    st.write("Answer: \n", actual_content)