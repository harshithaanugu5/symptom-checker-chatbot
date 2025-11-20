import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.chatbot import predict_condition

st.title("🩺 Symptom Checker Chatbot")

user_input = st.text_input("Describe your symptoms:")

if user_input:
    diagnosis = predict_condition(user_input)
    st.write(f"### Possible Condition: **{diagnosis}**")

    st.write("⚠️ This is not a medical diagnosis. Consult a doctor for professional advice.")
