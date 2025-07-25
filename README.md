# 💰 Financial Fraud Detection App

## Project Overview

This project focuses on developing a machine learning model to predict fraudulent transactions for a financial company. The goal is to provide a robust solution for proactive fraud detection, offering actionable insights to enhance security and prevent financial losses. The core of this project is a Streamlit web application that allows for interactive predictions based on the developed model.

## Problem Statement

Financial companies constantly face the challenge of identifying and preventing fraudulent transactions. With a massive volume of daily transactions, manual review is impractical. This project aims to build an automated system that can accurately flag suspicious transactions, enabling quicker response and better resource allocation for fraud investigation.

## Dataset

The model is developed using a comprehensive dataset containing financial transaction records.

* **Source:** [Insert Link to Data Source Here - e.g., the Kaggle link provided in your context]
* **Size:** 6,362,620 rows and 10 columns
* **Data Dictionary:** [Insert Link to Data Dictionary Here]

The dataset includes various features related to transactions, such as transaction amount, type, customer details, and a label indicating whether a transaction is fraudulent or not.

## Machine Learning Solution

### Data Preprocessing
The project involved standard data cleaning procedures, including:
* Handling missing values.
* Addressing outliers.
* Analyzing and mitigating multi-collinearity among features.
* Feature engineering (if applicable, e.g., creating new features from existing ones).

### Model Development
A machine learning model was developed and estimated on calibration data, then tested on validation data. The model is designed to classify transactions as either legitimate or fraudulent.

### Variable Selection
[**TO BE COMPLETED BY YOU:** Briefly describe your methodology for selecting variables for the model. For example: "Variables were selected based on correlation analysis, feature importance from tree-based models, and domain expertise to ensure relevance and predictive power." ]

## Streamlit Application Features

The interactive Streamlit application provides a user-friendly interface to:

* Input transaction details (or use default values for demonstration).
* Obtain a real-time prediction (Fraudulent / Not Fraudulent).
* View the confidence score (probability of fraud) associated with the prediction.
* Includes a disclaimer emphasizing the model's role as an assistive tool.

## How to Run the App (Locally)

While the app is deployed on Streamlit Community Cloud, you can also run it locally:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/jainanushk8/Fraud_Detection_Streamlit_App.git](https://github.com/jainanushk8/Fraud_Detection_Streamlit_App.git)
    cd Fraud_Detection_Streamlit_App
    ```
2.  **Install Dependencies:**
    Ensure you have Python (3.8 or higher recommended) and pip installed.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit App:**
    ```bash
    streamlit run streamlit_app.py
    ```
    The app will open in your web browser, typically at `http://localhost:8501`.

## Project Structure

.
├── streamlit_app.py          # Main Streamlit application script
├── requirements.txt          # Python dependencies
├── Fraud.csv                 # The dataset used for training/prediction
├── saved_model/              # Directory to store your trained ML model (e.g., .pkl, .h5)
│   └── fraud_model.pkl       # (Example) Your trained machine learning model file
└── README.md                 # This README file


* **Performance Metrics:**
    * Accuracy: `XX.X%`
    * Precision (Fraud): `XX.X%`
    * Recall (Fraud): `XX.X%`
    * F1-Score (Fraud): `XX.X%`
    * AUC-ROC Score: `X.XXX`
    *(Use the best set of tools to present these, e.g., confusion matrix visualizations, ROC curves).*

* **Key Factors Predicting Fraud:**
    * [List the top N features that predict fraud, e.g., "Transaction amount (particularly high values)", "Specific transaction types (e.g., Cash out)", "Number of previous failed transactions", etc.]

* **Factor Interpretation (Do these factors make sense?):**
    * [Explain why these factors logically make sense in the context of financial fraud. E.g., "High transaction amounts are often targeted by fraudsters to maximize gains," "Certain transaction types might be more vulnerable due to less stringent verification," etc.]

* **Prevention Strategies:**
    * [Describe the prevention measures recommended when the company updates its infrastructure (e.g., Two-factor authentication for certain transaction types, real-time anomaly detection, behavior analytics, stronger KYC checks for new accounts, implementing velocity checks).]

* **Determining Effectiveness:**
    * [Explain how you would measure if these implemented actions are working (e.g., Tracking reduction in fraud rate, monitoring false positive rate, A/B testing new prevention measures, analyzing customer friction, continuous model retraining and evaluation on new data).]

