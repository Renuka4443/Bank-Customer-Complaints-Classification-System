# 🏦 Bank Complaint Classification System  

## 💼 Overview  
The **Bank Complaint Classification System** is an **NLP-based solution** that automatically classifies customer complaints received by banks and financial institutions. It uses **machine learning models** trained on real-world complaint data to identify the appropriate product category, such as **Credit Card**, **Bank Account**, **Loans & Mortgages**, and others, making the complaint-handling process faster, more efficient, and improving the overall customer experience.  

### 🔑 Key Highlights  
- **Machine Learning Models Used:**  
   **1. Logistic Regression**  
   **2. Random Forest Classifier**  

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
