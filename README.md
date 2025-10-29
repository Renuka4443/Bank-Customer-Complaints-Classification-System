# 🏦 Bank Complaint Classification System  

## 💼 Overview  
The **Bank Complaint Classification System** is an **NLP-based solution** that automatically classifies customer complaints received by banks and financial institutions. It uses **machine learning models** trained on real-world complaint data to identify the appropriate product category, such as **Credit Card**, **Bank Account**, **Loans & Mortgages**, and others, making the complaint-handling process faster, more efficient, and improving the overall customer experience.  

### 🔑 Key Highlights  
- **Machine Learning Models Used:**  
  - **Logistic Regression**  
  - **Random Forest**  
- **Training Dataset:** Utilized the **CFPB Consumer Complaint Dataset** to learn complaint patterns across various banking products.  
- **Evaluation Dataset:** Tested the models on a **separate bank complaint dataset** to validate robustness and ensure generalization to unseen data.  

---

## 📂 Dataset Source  

This project uses two publicly available complaint datasets. The same preprocessing and TF-IDF feature extraction techniques were consistently applied to both datasets.  

| Dataset Type | Name & Link | Categories Included |
|---------------|-------------|----------------------|
| **Primary** | [Bank Customer Complaint Analysis](https://www.kaggle.com/datasets/adhamelkomy/bank-customer-complaint-analysis) | Credit Card, Retail Banking, Credit Reporting, Mortgages & Loans, Debt Collection |
| **Secondary** | [Consumer Complaints Data (CFPB)](https://www.kaggle.com/datasets/sebastienverpile/consumercomplaintsdata) | Debt Collection, Mortgage, Credit Reporting, Loan, Credit Card, Bank Account |

> Both datasets contain textual customer complaints along with corresponding product categories, enabling effective model training and evaluation across varied banking domains.  
