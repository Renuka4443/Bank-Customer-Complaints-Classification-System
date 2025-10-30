# 🏦 Bank Complaint Classification System  

## 💼 Overview  
The **Bank Complaint Classification System** is an **NLP-based solution** that automatically classifies customer complaints received by banks and financial institutions. It uses **machine learning models** trained on real-world complaint data to identify the appropriate product category, such as **Credit Card**, **Bank Account**, **Loans & Mortgages**, and others making the complaint-handling process faster, more efficient, and improving the overall customer experience.  

---

### 🔑 Key Highlights  
- **Machine Learning Models Used:**  
   1. Logistic Regression  
   2. Random Forest Classifier  

- **Training & Evaluation Datasets:**  
  The models were trained and evaluated on **two similar real-world datasets**, the **Bank Customer Complaint Dataset** and the **CFPB Consumer Complaint Dataset**, to compare model performance and assess generalization across different data sources.  

---

## 📂 Dataset Source  

This project uses two publicly available complaint datasets. The same preprocessing steps and **TF-IDF feature extraction** techniques were consistently applied to both datasets to ensure uniformity and comparability.  

| Dataset Type | Name & Link | Number of Records | Categories Included |
|---------------|-------------|-------------------|----------------------|
| **Dataset 1 (Primary)** | [Bank Customer Complaint Analysis](https://www.kaggle.com/datasets/adhamelkomy/bank-customer-complaint-analysis) | **162,421** | Credit Reporting, Debt Collection, Credit Card, Mortgages & Loans, Retail Banking |
| **Dataset 2 (Secondary)** | [Consumer Complaints Data (CFPB)](https://www.kaggle.com/datasets/sebastienverpile/consumercomplaintsdata) | **24,374** | Debt Collection, Mortgage, Credit Reporting, Loan, Credit Card, Bank Account |

> Both datasets contain textual customer complaints along with corresponding product categories, enabling effective **training**, **validation**, and **performance comparison** of machine learning models across varied banking domains.  

---

## ⚙️ Methodology  

The project follows a structured NLP-based machine learning pipeline for efficient and accurate complaint classification.  
Below is the end-to-end workflow followed in this project:

<p align="center">
  <img src="https://github.com/user-attachments/assets/27a7c095-cc8d-47bf-959d-f687e67fafca" alt="Methodology Flow Diagram" width="1000">
</p>

1. **Data Collection:**  
   Gathered complaint data from two real-world sources — the **Bank Customer Complaint Dataset** and the **CFPB Consumer Complaint Dataset**.  

2. **Data Preprocessing:**  
   Removed null values, duplicates, punctuation, and stopwords; converted text to lowercase and applied lemmatization for normalization, followed by addressing class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)** to ensure fair model learning across all complaint categories.  

3. **Feature Extraction:**  
   Transformed complaint text into numerical vectors using **TF-IDF** for machine learning model input.  

4. **Model Training:**  
   Implemented and compared **Logistic Regression** and **Random Forest Classifier** for multi-class text classification.  

5. **Evaluation:**  
   Measured model performance using **Accuracy**, **Precision**, **Recall**, and **F1-score**.  

6. **Model Validation:**  
   Conducted **cross-dataset evaluation** to test robustness and generalization across two independent complaint datasets.  

7. **User Input & Real-Time Prediction:**  
   Integrated the trained model into a simple frontend interface where users can **enter a complaint** and instantly receive the **predicted product category**, demonstrating real-time classification capabilities.  
