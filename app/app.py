# ================================
# Imports
# ================================

import streamlit as st
import joblib
import re
import string
import pdfplumber
import numpy as np


# ================================
# Load Models
# ================================

model = joblib.load("../output/model.pkl")
vectorizer = joblib.load("../output/vectorizer.pkl")
encoder = joblib.load("../output/encoder.pkl")


# ================================
# Text Cleaning (Same as training)
# ================================

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ================================
# PDF Text Extraction (NO OCR)
# ================================

def extract_text_from_pdf(uploaded_file):
    text = ""
    
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    
    return text


# ================================
# Streamlit UI
# ================================

st.set_page_config(page_title="Resume Analyzer", layout="centered")

st.markdown("# Resume Analyzer")
st.markdown("### NLP-based Job Category Prediction System")

st.markdown("---")


# ================================
# Input Mode Selection
# ================================

input_mode = st.radio(
    "Choose Input Method",
    ["Paste Resume Text", "Upload PDF"],
    horizontal=True
)


resume_text = ""


# ================================
# Text Input
# ================================

if input_mode == "Paste Resume Text":
    resume_text = st.text_area(
        "Paste your resume below",
        height=300,
        placeholder="Enter resume content..."
    )


# ================================
# PDF Upload
# ================================

elif input_mode == "Upload PDF":
    uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            resume_text = extract_text_from_pdf(uploaded_file)
        
        if resume_text.strip() == "":
            st.error("No readable text found in PDF. Please upload a text-based resume.")
        else:
            st.success("Text extracted successfully")
            st.text_area("Preview (first 1000 characters)", resume_text[:1000], height=200)


st.markdown("---")


# ================================
# Prediction
# ================================

if st.button("Analyze Resume"):
    
    if resume_text.strip() == "":
        st.warning("Please provide resume input")
    
    else:
        # Clean text
        cleaned_text = clean_text(resume_text)
        
        # Vectorize
        vectorized = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(vectorized)
        probabilities = model.predict_proba(vectorized)
        
        predicted_label = encoder.inverse_transform(prediction)[0]
        confidence = np.max(probabilities)
        
        # Output
        st.markdown("## Prediction Result")
        st.success(f"Predicted Category: {predicted_label}")
        st.info(f"Confidence Score: {confidence:.2f}")
        
        st.caption("Model: Random Forest | Features: TF-IDF")