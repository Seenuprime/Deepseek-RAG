from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
import streamlit as st
import fitz
import os
from dotenv import load_dotenv
load_dotenv()

st.title("🚀 Welcome to DeepSeek RAG - Powered by ChatGroq!")
st.write("Retrieve. Analyze. Generate - Smarter AI Answers!")

api = st.sidebar.text_input("Please enter your Groq API Key here:", type='password')
if api:
    llm = ChatGroq(model='deepseek-r1-distill-qwen-32b', api_key=api)

    # # Load the text file and chunk the document
    # loader = TextLoader('data.txt')
    # docs = loader.load()

    # Load the Pdf file and chunk the document
    # pdf_loader = PyMuPDFLoader('attention_all_youneed.pdf')
    # docs = pdf_loader.load()

    ## loading directly from the web
    file_loaded = st.sidebar.file_uploader("Choose the PDF file", type="pdf", accept_multiple_files=False)

    if file_loaded:
        st.sidebar.write(f"{file_loaded.name} loaded successfully!")

        file = fitz.open(file_loaded)
        text = ""
        for page in file:
            text += page.get_text()

        # Split the docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separators='/n')
        text_split = text_splitter.split_text(text)
        chunks = text_splitter.create_documents(text_split)

        # Initialize embedding and store in FAISS vector StopIteration
        embedding_model = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_store = FAISS.from_documents(chunks, embedding_model)

        prompt_template = PromptTemplate.from_template(
           """
                You are a highly intelligent AI assistant with access to the following retrieved knowledge. 
                Your task is to **provide clear, concise, and well-structured answers** based on the given context.\
                The user can reffer context as file, pdf and context all these are same.

                **Guidelines:**
                - Present answers in a **professional and engaging** manner.
                - Use **bullet points, paragraphs, or bold text** to improve readability.
                - If the answer **is not found in the context**, simply say:  
                *"I couldn't find the answer in the provided information."*
                - However, if you can provide an answer based on general knowledge, say:  
                *"Here is what I know:"* and then answer.

                **Retrieved Context:**
                ---
                {context}
                ---

                **User Question:**  
                {query}

                **Your Response:**  
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
                # print(think_content)
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