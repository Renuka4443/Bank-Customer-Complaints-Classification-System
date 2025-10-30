# 🏦 Bank Complaint Classification System

## 💼 Overview  
The **Bank Complaint Classification System** is an **NLP-based solution** that automatically classifies customer complaints received by banks and financial institutions. It uses **machine learning models** trained on real-world complaint data to identify the appropriate product category, such as **Credit Card**, **Bank Account**, **Loans & Mortgages**, and others making the complaint-handling process faster, more efficient, and improving the overall customer experience.  

### 🔑 Key Highlights  
- **Machine Learning Models Used:**  
   1. Logistic Regression  
  2. Support Vector Machine (LinearSVC)

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


### Distribution of Complaints

<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/8186cc61-69ac-40f0-b358-1c957631d53b" width="400" alt="Dataset 1 Distribution" />
      <br>
      <b>Figure 1:</b> Dataset 1 Distribution
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b23b3ff0-c740-4272-b76b-9e5a315ecff9" width="400" alt="Dataset 2 Distribution" />
      <br>
      <b>Figure 2:</b> Dataset 2 Distribution
    </td>
  </tr>
</table>



---

## ⚙️ Methodology  

The project follows a structured NLP-based machine learning pipeline for efficient and accurate complaint classification.  
Below is the end-to-end workflow followed in this project:

<p align="center">
  <img src="https://github.com/user-attachments/assets/d72d235b-ed23-404c-9f37-38e242b7d0c1" alt="Methodology Flow Diagram" width="1000">
</p>

1. **Data Collection:**  
   Gathered complaint data from two real-world sources: the **Bank Customer Complaint Dataset** and the **CFPB Consumer Complaint Dataset**.  

2. **Data Preprocessing:**  
   Removed null values, duplicates, punctuation, and stopwords; converted text to lowercase and applied lemmatization for normalization, followed by addressing class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)** to ensure fair model learning across all complaint categories.  

3. **Feature Extraction:**  
   Transformed complaint text into numerical vectors using **TF-IDF** for machine learning model input.  

4. **Model Training:**  
  Implemented and compared **Logistic Regression** and **Support Vector Machine (LinearSVC)** for multi-class text classification.

5. **Evaluation:**  
   Measured model performance using **Accuracy**, **Precision**, **Recall**, and **F1-score**.  

6. **Model Validation:**  
   Conducted **cross-dataset evaluation** to test robustness and generalization across two independent complaint datasets.  

7. **User Input & Real-Time Prediction:**  
   Integrated the trained model into a simple frontend interface where users can **enter a complaint** and instantly receive the **predicted product category**, demonstrating real-time classification capabilities.

   
---

## 🧩 Model Selection Rationale  

Two models were chosen for this project: **Logistic Regression** and **Support Vector Machine (LinearSVC)** due to their proven effectiveness in text classification tasks.  

**Logistic Regression** works well with TF-IDF features, is simple, fast, and interpretable, making it a strong baseline for multi-class text data.  

**LinearSVC** performs efficiently on high-dimensional, sparse data, and provides higher precision and recall by finding the optimal separating hyperplane between complaint categories.  

Both models balance accuracy, speed, and scalability, making them ideal for real-time complaint classification.  

---

## 📈 Experiments & Results Summary  

Both models were trained and evaluated on the two datasets using Accuracy, Precision, Recall, and F1-Score to assess overall performance and generalization ability.

<img width="1200" height="400" alt="Screenshot 2025-10-31 005018" src="https://github.com/user-attachments/assets/ffbcd6d2-045e-4652-8fac-0ecd00a9a3a2" />

### Dataset 1: Bank Customer Complaint Dataset  

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|----------|-----------|
| Logistic Regression | 0.8615 | 0.8717 | 0.8615 | 0.8644 |
| Support Vector Machine (LinearSVC) | 0.8545 | 0.8648 | 0.8545 | 0.8576 |

### Dataset 2: CFPB Consumer Complaint Dataset  

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|----------|-----------|
| Logistic Regression | 0.8554 | 0.8563 | 0.8554 | 0.8557 |
| Support Vector Machine (LinearSVC) | 0.8480 | 0.8486 | 0.8480 | 0.8482 |

### 🔍 Observations  

- **Logistic Regression** slightly outperformed **LinearSVC** on both datasets, demonstrating better generalization and stability.  
- **LinearSVC** showed competitive performance, proving effective for large-scale, sparse text data.  
- The consistency of results across both datasets indicates that the models are robust and reliable for real-world complaint classification.

---

## 🚀 Steps to Run the Project

Follow these simple steps to set up and run the project locally on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/Renuka4443/Bank-Complaint-Classification-System
cd Bank-Complaint-Classification-System
```
### 2. (Optional) Create and Activate a Virtual Environment

**For Windows (PowerShell):**

```bash
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```
**For macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```
### 4. Run the Application

```bash
streamlit run app.py
```
Once the app starts, open the **local URL** shown in your terminal to access the **Bank Complaint Classification System**.

## 🛠️ Technologies Used  

<div align="center">

<table>
<tr>
<td align="center" width="220">
  <a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  </a>
</td>
<td align="center" width="220">
  <a href="https://streamlit.io/" target="_blank">
    <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
</td>
<td align="center" width="220">
  <a href="https://scikit-learn.org/" target="_blank">
    <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  </a>
</td>
<td align="center" width="220">
  <a href="https://imbalanced-learn.org/" target="_blank">
    <img src="https://img.shields.io/badge/imbalanced--learn-SMOTE-005571?style=for-the-badge" />
  </a>
</td>
</tr>
<tr>
<td align="center" width="220">
  <a href="https://www.nltk.org/" target="_blank">
    <img src="https://img.shields.io/badge/NLTK-NLP-154F28?style=for-the-badge" />
  </a>
</td>
<td align="center" width="220">
  <a href="https://pandas.pydata.org/" target="_blank">
    <img src="https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  </a>
  <br/>
  <a href="https://numpy.org/" target="_blank">
    <img src="https://img.shields.io/badge/NumPy-Array-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  </a>
</td>
<td align="center" width="220">
  <a href="https://plotly.com/python/" target="_blank">
    <img src="https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  </a>
</td>
<td align="center" width="220">
  <a href="https://jupyter.org/" target="_blank">
    <img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  </a>
</td>
</tr>
</table>

</div>



---

## 🏁 Conclusion

The Bank Complaint Classification System demonstrates how **machine learning (ML)** and **natural language processing (NLP)** can automate the classification of customer complaints with high accuracy and reliability. By leveraging models such as **Logistic Regression** and **LinearSVC**, the system streamlines complaint handling, improves operational efficiency, and promotes data-driven decision-making within banking services.

---
## 📚 References
1. Adham Elkomy. *Bank Customer Complaint Analysis Dataset*. Kaggle, 2022. [Link](https://www.kaggle.com/datasets/adhamelkomy/bank-customer-complaint-analysis)  
2. Sebastien Verpile. *Consumer Complaints Data (CFPB)*. Kaggle, 2023. [Link](https://www.kaggle.com/datasets/sebastienverpile/consumercomplaintsdata)  


