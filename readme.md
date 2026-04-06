# Resume Analyzer using NLP & Machine Learning

## Overview

This project implements a complete **Resume Analyzer system** using Natural Language Processing (NLP) and Machine Learning techniques. The system processes raw resume text and classifies it into predefined job categories.

The project follows a structured pipeline:

* NLP preprocessing of resume text
* Feature extraction using TF-IDF
* Training multiple machine learning models
* Model evaluation and comparison
* Selection and saving of the best-performing model

---

## Problem Statement

Given a resume, the goal is to automatically predict its **job category** (e.g., HR, Engineering, Finance, IT, etc.).

This is a:

* Supervised Learning problem
* Multi-class Classification problem

---

## Dataset

The dataset contains **2484 resumes** across **24 job categories**.

### Columns:

* `Final_Resume` → Cleaned resume text (input feature)
* `Category` → Job category (target label)

---

## Project Structure

```
ml project/
│
├── resume.csv
├── resume_cleaned.csv
│
├── notebook1_nlp.ipynb        # NLP preprocessing pipeline
├── notebook2_ml.ipynb         # ML training and evaluation
│
└── output/
    ├── model.pkl
    ├── vectorizer.pkl
    └── encoder.pkl
```

---

## NLP Pipeline (Notebook 1)

The raw resume text is processed using the following steps:

1. Text Cleaning

   * Lowercasing
   * Removal of URLs, punctuation, and numbers

2. Tokenization

   * Splitting text into words

3. Stopword Removal

   * Removing non-informative words

4. Lemmatization

   * Converting words to base form

5. Reconstruction

   * Converting processed tokens back to text

Output:

```
resume_cleaned.csv
```

---

## Feature Extraction

TF-IDF (Term Frequency – Inverse Document Frequency) is used to convert text into numerical vectors.

* Max features: 5000
* Sparse high-dimensional representation

---

## Machine Learning Models

The following models were trained and evaluated:

* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Decision Tree (CART - Gini Index)
* Artificial Neural Network (ANN)

---

## Evaluation Metrics

Each model was evaluated using:

* Accuracy
* Precision (Weighted)
* Recall (Weighted)
* F1 Score (Weighted)
* Specificity
* Confusion Matrix

---

## Overfitting Analysis

| Model         | Behavior     |
| ------------- | ------------ |
| Decision Tree | Overfitting  |
| ANN           | Overfitting  |
| SVM           | Balanced     |
| KNN           | Underfitting |

---

## Final Model Selection

**Support Vector Machine (SVM)** was selected as the best model due to:

* Balanced bias-variance tradeoff
* Strong performance on high-dimensional TF-IDF data
* Better generalization on unseen data

---

## Saved Artifacts

The following files are saved for deployment:

* `model.pkl` → Trained SVM model
* `vectorizer.pkl` → TF-IDF transformer
* `encoder.pkl` → Label encoder

---

## How It Works

```
Input Resume
    ↓
Text Cleaning (NLP)
    ↓
TF-IDF Vectorization
    ↓
SVM Model Prediction
    ↓
Decoded Category Output
```

---

## Key Concepts Covered

* Natural Language Processing (NLP)
* TF-IDF Vectorization
* SVM, KNN, Decision Trees, ANN
* Overfitting and Underfitting
* Precision, Recall, F1 Score
* Confusion Matrix
* Model Deployment using Pickle

---

## Future Work

* Build Streamlit web application for real-time prediction
* Improve accuracy using advanced models (e.g., BERT)
* Add skill extraction and ranking
* Implement resume-job matching

---

## Author

Mandeep Singh
Shubham Raj

---

## Conclusion

This project demonstrates a complete end-to-end machine learning pipeline for text classification, combining NLP preprocessing with multiple classification algorithms to build an effective Resume Analyzer system.
