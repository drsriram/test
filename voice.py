import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from gtts import gTTS
import io
import base64


models = [
    "llama-3.1-70b-versatile",
    "Mixtral-8x7B-32768",
    "llama-3.1-8b-instant",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "gemma-7b-it"
]

st.title("Email Classification with Groq API and Voice Commands")
st.write("Classify email responses into predefined categories and hear the response.")

api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    selected_model = st.selectbox("Select a model:", models)
    llm = ChatGroq(groq_api_key=api_key, model_name=selected_model)

    system_prompt = (
        "You are an assistant for email classification tasks. "
        "Classify the email response into one of the following categories: "
        "1. Right contact (category 1) "
        "2. Not the right contact (category 2) "
        "3. Right contact & Name change (category 3) "
        "4. Right contact & Add another contact name (category 4) "
        "5. Uncertain/Other (category 5). "
        "Provide a concise classification based on the content of the email response."
    )

    classification_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="input"),
        ]
    )

    user_input = st.text_input("Your email response:")
    if user_input:
        formatted_prompt = classification_prompt.format_messages(input=[HumanMessage(content=user_input)])

        response = llm.invoke(formatted_prompt)
        classification_result = response.content
        st.write("Assistant:", classification_result)

        tts = gTTS(text=classification_result, lang='en')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)

        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
        audio_url = f"data:audio/mp3;base64,{audio_base64}"

        st.audio(audio_url, format="audio/mp3")

else:
    st.warning("Please enter the Groq API Key")

#i want this comment and write this comment in local repo
