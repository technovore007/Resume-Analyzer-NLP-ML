# Resume Analyzer using NLP and Machine Learning

---

## Overview

This project implements an end-to-end **Resume Analyzer system** using Natural Language Processing (NLP) and Machine Learning techniques. The system processes raw resume text and classifies it into predefined job categories such as HR, Engineering, Finance, and Information Technology.

---

## System Workflow

![Workflow](images/workflow.png)

---

## Problem Statement

Given a resume, the objective is to automatically predict its corresponding job category.

This is a:

* Supervised Learning problem
* Multi-class Classification problem

---

## Dataset

* Total resumes: **2484**
* Total categories: **24**

The dataset used in this project is publicly available on Kaggle:

**Resume Dataset**
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/data

### Features

* `Final_Resume` — Cleaned resume text
* `Category` — Target label

---

## Project Structure

```
ml-project/
│
├── resume.csv
├── resume_cleaned.csv
│
├── notebook1_nlp.ipynb        # NLP preprocessing pipeline
├── ResumeAnalyzer_ML.ipynb         # ML training and evaluation
│
├── output/
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── encoder.pkl
│
└── images/
    ├── workflow.png
    ├── confusion_matrix.png
    └── results.png
```

---

## NLP Pipeline

The raw resume text is processed through the following steps:

1. Text Cleaning

   * Lowercasing
   * Removal of URLs, punctuation, and numbers

2. Tokenization

3. Stopword Removal

4. Lemmatization

5. Text Reconstruction

Output:

```
resume_cleaned.csv
```

---

## Feature Extraction

TF-IDF (Term Frequency – Inverse Document Frequency) is used to convert text into numerical vectors.

* Maximum features: **5000**
* Produces a sparse, high-dimensional representation

---

## Machine Learning Models

The following models were trained and evaluated:

* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Decision Tree (CART with Gini Index)
* Artificial Neural Network (ANN)

---

## Model Evaluation

![Confusion Matrix](images/confusion_matrix.png)

Evaluation metrics:

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

Support Vector Machine (SVM) was selected as the final model due to:

* Balanced bias-variance tradeoff
* Strong performance on high-dimensional TF-IDF features
* Better generalization on unseen data

---

## Saved Artifacts

The following components are saved for deployment:

* `model.pkl` — Trained SVM model
* `vectorizer.pkl` — TF-IDF transformer
* `encoder.pkl` — Label encoder

---

## Inference Pipeline

```
Input Resume
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
SVM Prediction
    ↓
Category Output
```

---

## Key Concepts Covered

* Natural Language Processing (NLP)
* TF-IDF Vectorization
* SVM, KNN, Decision Trees, ANN
* Overfitting and Underfitting
* Precision, Recall, F1 Score
* Confusion Matrix
* Model Serialization (Pickle)

---

## Future Work

* Develop a Streamlit-based web application

---

## Authors

* Mandeep Singh
* Shubham Raj

---

## License

This project is licensed under the MIT License.
See the LICENSE file for details.

---

## Conclusion

This project demonstrates a complete machine learning pipeline for text classification, combining NLP preprocessing and multiple classification models to build an effective and extensible Resume Analyzer system.
