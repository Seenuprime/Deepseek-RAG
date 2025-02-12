from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

st.title("Welcome to DeepSeek integrated with ChatGroq")

api = st.sidebar.text_input("Please enter your Groq API Key here:", type='password')
if api:
    llm = ChatGroq(model='deepseek-r1-distill-qwen-32b', api_key=api)

query = st.text_input("Enter you query: ")

if query and api:
    prompt_template = PromptTemplate.from_template("""You are a science expert. Answer concisely with examples, 
                                               and You are a logical AI. Explain step by step.
                                               Explain {topic} in simple terms with examples.
                                               you can use emojies and make it look good.
                                                give long answers if needed.""")
    
    prompt = prompt_template.format(topic=query)
    
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