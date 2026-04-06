# ================================
# Step 1: Imports
# ================================

import streamlit as st
import joblib
import re
import string


# ================================
# Step 2: Load Saved Artifacts
# ================================

model = joblib.load("../output/model.pkl")
vectorizer = joblib.load("../output/vectorizer.pkl")
encoder = joblib.load("../output/encoder.pkl")


# ================================
# Step 3: Text Cleaning Function
# (Same as Notebook NLP)
# ================================

def clean_text(text):
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ================================
# Step 4: Streamlit UI Setup
# ================================

st.set_page_config(page_title="Resume Analyzer", layout="centered")

st.title("Resume Analyzer")
st.write("Paste a resume below to predict its job category")


# ================================
# Step 5: Input Box
# ================================

resume_text = st.text_area("Enter Resume Text")


# ================================
# Step 6: Button (No logic yet)
# ================================

# ================================
# Step 7: Prediction Logic
# ================================

if st.button("Analyze Resume"):
    
    if resume_text.strip() == "":
        st.warning("Please enter resume text")
    
    else:
        # Clean text
        cleaned_text = clean_text(resume_text)
        
        # Vectorize
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(vectorized_text)
        
        # Decode label
        predicted_category = encoder.inverse_transform(prediction)[0]
        
        # Display result
        st.success(f"Predicted Category: {predicted_category}") 