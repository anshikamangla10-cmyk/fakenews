#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Load Models and Vectorizer ---
LR = joblib.load("logistic_model.pkl")
DT = joblib.load("decision_tree_model.pkl")
GB = joblib.load("gradient_boost_model.pkl")
RF = joblib.load("random_forest_model.pkl")
vectorization = joblib.load("tfidf_vectorizer.pkl")

# --- Text Cleaning Function ---
def wordopt(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# --- Label Mapping ---
def output_label(n):
    return "🟥 Fake News" if n == 0 else "🟩 Not a Fake News"

# --- Streamlit UI ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article below to check if it's **fake or real** using multiple ML models.")

news_input = st.text_area("📝 Paste News Article Here", height=200, placeholder="Type or paste news content...")

if st.button("Detect"):
    if news_input.strip() == "":
        st.warning("Please enter some news content to analyze.")
    else:
        cleaned_text = wordopt(news_input)
        vectorized_text = vectorization.transform([cleaned_text])

        pred_LR = LR.predict(vectorized_text)[0]
        pred_DT = DT.predict(vectorized_text)[0]
        pred_GBC = GB.predict(vectorized_text)[0]
        pred_RFC = RF.predict(vectorized_text)[0]

        st.subheader("🔍 Model Predictions")
        st.markdown(f"**Logistic Regression:** {output_label(pred_LR)}")
        st.markdown(f"**Decision Tree:** {output_label(pred_DT)}")
        st.markdown(f"**Gradient Boosting:** {output_label(pred_GBC)}")
        st.markdown(f"**Random Forest:** {output_label(pred_RFC)}")

        st.markdown("---")
        st.info("These predictions are based on trained models using TF-IDF features and cleaned news data.")

