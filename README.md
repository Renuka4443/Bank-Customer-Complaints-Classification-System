# 🏦 Bank Complaint Classification System

## 💼 Overview  

The **Bank Complaint Classification System** is an **NLP-based application** that categorizes the **customer complaints** that are sent to **banks and financial services**. It makes use of **machine learning models** built on **actual complaint data** to determine the **relevant product**, for instance **Credit Card**, **Bank Account**, **Loans & Mortgages** and others, helping the **customer service** to deal with the complaints **quicker**, **better**, and the **customer experience** to be still **evolved**.  


### 🔑 Key Highlights  
- **Machine Learning Algorithms Applied:**  
   1. Logistic Regression  
  2. Support Vector Machine (LinearSVC)

- **Training & Evaluation Datasets:**  
  The models were trained and tested on two closely related real-world datasets, namely the **Bank Customer Complaint Dataset** and the **CFPB Consumer Complaint Dataset**, to analyze the performance of the models and test their generalization to a different source of data.  


---

## 📂 Dataset Source  

This project uses two publicly available complaint datasets. The same preprocessing steps and **TF-IDF feature extraction** techniques were consistently applied to both datasets to ensure uniformity and comparability.  

| Dataset Type | Name & Link | Number of Records | Categories Included |
|---------------|-------------|-------------------|----------------------|
| **Dataset 1 (Primary)** | [Bank Customer Complaint Analysis](https://www.kaggle.com/datasets/adhamelkomy/bank-customer-complaint-analysis) | **162,421** | Credit Reporting, Mortgages & Loans, Debt Collection, Retail Banking, Credit Card |
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


### **1. Data Collection**
The complaint data was collected from two sources of real-world data: the **Bank Customer Complaint Dataset** and the **CFPB Consumer Complaint Dataset**.  

### **2. Preprocessing Data**
Preprocessing is mainly concerned with **cleaning** and **normalizing** the complaint text before providing it to the machine learning models.  
The following steps were taken: 

1. **Removing Null Values and Duplicates:** Dropped all the rows/columns that had null values relating to the column's integrity.  
2. **Text Normalization:** Converted all text to lowercase for uniformity.  
3. **Noise Removal:** Removed punctuations, special characters, and numbers that are not useful for classification.  
4. **Stopword Removal:** Removed common English stopwords like *the*, *is*, and *are*, etc., based on the **NLTK stopword list**.  
5. **Tokenization:** Broke sentences into words (tokens) for performing fine-grained text analytics.  
6. **Lemmatization:** Transformed words into their base or dictionary form (e.g., “customers” → “customer”).  
7. **Addressing Class Imbalance:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the underrepresented complaint classes.  

### **3. Feature Extraction**
Implemented **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert the complaint text into **numerical feature vectors** for the machine learning model.  

### **4. Model Training**
Implemented and compared **Logistic Regression** and **Support Vector Machine (LinearSVC)** for **multi-class text classification**.  

### **5. Model Evaluation**
Evaluated the model’s performance using **Accuracy**, **Precision**, **Recall**, and **F1-Score** metrics.  

### **6.Model Validation**
Performed **cross-dataset validation** to verify model **robustness** and **generalizability** using the two independent complaint datasets.  

### **7. User Input & Real-Time Prediction**
The trained model was integrated into a **simple front-end interface**, allowing users to input a complaint and receive an **instant prediction** of the product category, showcasing the ability to deliver **real-time classification**.  



   
---

## 🧩 Model Selection Rationale  

Two models were selected for the project: **Logistic Regression** and **Support Vector Machine (LinearSVC)**, as they have been proven to be highly effective for **text classification problems**.  

- **Logistic Regression** performs well with **TF-IDF features**, and its **simplicity, speed, and interpretability** make it a strong baseline for **multi-class text data**.  

- **LinearSVC** efficiently handles **high-dimensional sparse data** and achieves even higher **precision** and **recall** by learning an **optimal separating hyperplane** between the categories of complaints.  

Both models effectively balance **accuracy**, **speed**, and **scalability**, making them ideal for **real-time complaint classification** in the banking domain.  
 

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
- **Logistic Regression** achieved a better result than the **LinearSVC** in both datasets, indicating it was more stable and generalized better.  
- **LinearSVC** was also very competitive, working well on large-scale sparse text data.  
- The uniformly good performance on both datasets suggests that the models are mature and can be effectively used to classify real-world complaints.

---

## 🚀 Steps to Run the Project

Set up and execute the project locally on your PC by following these instructions:

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

---


### 🌐 Live Demo  

You can access the **live demo** of the project here:  
👉 [**Bank Complaint Classification System - Live Demo**](https://bank-customer-complaints-classification-system.streamlit.app/)  

---



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

The **Bank Complaint Classification System** illustrates the usage of **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques for accurate and reliable automated classification of customer complaints. By employing models like **Logistic Regression** and **LinearSVC** for complaint classification, the system enhances the complaint management process by increasing **operational efficiency** and enabling **data-driven decisions** in the banking service areas.


---
## 📚 References
1. Adham Elkomy. *Bank Customer Complaint Analysis Dataset*. Kaggle, 2022. [Link](https://www.kaggle.com/datasets/adhamelkomy/bank-customer-complaint-analysis)  
2. Sebastien Verpile. *Consumer Complaints Data (CFPB)*. Kaggle, 2023. [Link](https://www.kaggle.com/datasets/sebastienverpile/consumercomplaintsdata)  




