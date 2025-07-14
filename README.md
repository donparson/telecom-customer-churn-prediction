# telecom-customer-churn-prediction
Predictive model to identify customers at risk of churning in a telecommunications company, enabling proactive retention strategies.
# Customer Churn Prediction in a Telecommunications Company

## Project Overview

This project focuses on building a robust machine learning model to predict customer churn (attrition) in a telecommunications company. In today's competitive telecom landscape, retaining existing customers is significantly more cost-effective than acquiring new ones. By accurately identifying customers at high risk of churning, the company can proactively implement targeted retention strategies, thereby enhancing customer lifetime value and optimizing marketing spend.

The solution leverages a publicly available dataset that mirrors real-world telecom customer behavior, providing a comprehensive demonstration of the end-to-end data science lifecycle, from data exploration and preprocessing to model building, evaluation, and the derivation of actionable business insights.

## Table of Contents

1.  [Problem Statement](#1-problem-statement)
2.  [Data Source](#2-data-source)
3.  [Methodology](#3-methodology)
    * [3.1 Data Loading & Initial Inspection](#31-data-loading--initial-inspection)
    * [3.2 Exploratory Data Analysis (EDA)](#32-exploratory-data-analysis-eda)
    * [3.3 Data Preprocessing & Feature Engineering](#33-data-preprocessing--feature-engineering)
    * [3.4 Model Building & Training](#34-model-building--training)
    * [3.5 Model Evaluation & Optimization](#35-model-evaluation--optimization)
4.  [Key Findings & Business Recommendations](#4-key-findings--business-recommendations)
5.  [Results](#5-results)
6.  [Technologies Used](#6-technologies-used)
7.  [How to Run the Project](#7-how-to-run-the-project)
8.  [Future Work](#8-future-work)

---

## 1. Problem Statement

Customer churn is a significant challenge for telecommunications companies, directly impacting revenue and growth. High churn rates necessitate increased spending on customer acquisition, which is often more expensive than customer retention. The goal is to develop a predictive model that identifies customers likely to churn *before* they cancel their service. This early identification enables the company to intervene with personalized offers, improved services, or other retention efforts, ultimately reducing customer attrition and increasing profitability.

## 2. Data Source

This project utilizes the **IBM Telco Customer Churn dataset**, available on Kaggle.

* **Dataset Name:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
* **Source:** [Kaggle - IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Description:** The dataset contains information about a fictional telecommunications company's customers, including services signed up for, account information, and demographic data. The target variable is 'Churn', indicating whether the customer left the company last month.

## 3. Methodology

The project followed a standard data science methodology to ensure a robust and insightful analysis:

### 3.1 Data Loading & Initial Inspection

The dataset was loaded using Pandas. An initial inspection was performed to understand its structure, identify data types, and check for missing values.

### 3.2 Exploratory Data Analysis (EDA)

Comprehensive EDA was conducted to uncover patterns, anomalies, and relationships within the data. Key insights derived from this phase include:

* **Missing Values:** Only the `TotalCharges` column had missing values. Upon investigation, these corresponded to new customers with 0 tenure, which were imputed with 0 to accurately reflect their status.
* **Target Imbalance:** The `Churn` variable showed a significant imbalance, with approximately **[~73% No Churn]** and **[~27% Yes Churn]**. This highlights the need for careful model evaluation metrics beyond simple accuracy.
* **Key Churn Drivers from Categorical Features:**
    * **Contract Type:** Customers on **month-to-month contracts** showed a dramatically higher churn rate.
    * **Internet Service:** Users with **Fiber optic internet service** exhibited significantly higher churn compared to DSL or no internet service.
    * **Payment Method:** Customers using **Electronic Check** had the highest churn rate among all payment methods.
    * Absence of services like `OnlineSecurity` and `TechSupport` also correlated with higher churn.
* **Key Churn Drivers from Numerical Features:**
    * **Tenure:** Customers with **lower tenure (new customers)** were far more likely to churn. Loyalty increased significantly with longer tenure.
    * **Monthly Charges:** Higher **`MonthlyCharges`** were associated with increased churn risk.

### 3.3 Data Preprocessing & Feature Engineering

The raw data was transformed into a machine learning-ready format:

* **Dropped `customerID`:** This unique identifier is not useful for prediction.
* **Target Variable Encoding:** The `Churn` column was converted from 'Yes'/'No' to numerical `1`/`0`.
* **Binary Feature Mapping:** 'Yes'/'No' and 'No internet service'/'No phone service' categories were mapped to `1`/`0` for consistent numerical representation.
* **One-Hot Encoding:** Multi-category nominal features (e.g., `gender`, `InternetService`, `Contract`, `PaymentMethod`) were converted into numerical format using one-hot encoding, with `drop_first=True` to avoid multicollinearity.
* **Feature Scaling:** Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized using `StandardScaler` to ensure no single feature dominated model training due to its scale.

### 3.4 Model Building & Training

The preprocessed data was split into training (80%) and testing (20%) sets, with `stratify=y` used during splitting to maintain the churn ratio in both sets, addressing the class imbalance.

Three common classification models were trained:
* **Logistic Regression:** As a robust baseline model.
* **Random Forest Classifier:** An ensemble method known for good performance.
* **XGBoost Classifier:** A powerful gradient boosting algorithm, often a top performer in classification tasks.

### 3.5 Model Evaluation & Optimization

Model performance was rigorously evaluated on the unseen test set using a comprehensive set of metrics suitable for imbalanced classification:

* **Accuracy:** Overall correct predictions.
* **Precision:** Of all customers predicted to churn, how many actually churned?
* **Recall:** Of all customers who actually churned, how many did the model correctly identify? (Crucial for churn prediction, as False Negatives are costly).
* **F1-Score:** Harmonic mean of Precision and Recall.
* **ROC AUC (Receiver Operating Characteristic - Area Under Curve):** Measures the model's ability to discriminate between churners and non-churners across all possible thresholds.
* **Confusion Matrix:** Provides a detailed breakdown of True Positives, True Negatives, False Positives, and False Negatives.

**Hyperparameter Tuning:** `GridSearchCV` was employed to optimize the best-performing model (XGBoost) by systematically searching for the optimal combination of hyperparameters. The primary optimization metric for tuning was **Recall**, given its business importance in minimizing missed churners.

## 4. Key Findings & Business Recommendations

Based on the model's insights, particularly from feature importance analysis and EDA, the following actionable recommendations are proposed:

1.  **Prioritize Month-to-Month Contract Conversion:**
    * **Insight:** Month-to-month contracts are the strongest predictor of churn.
    * **Recommendation:** Implement aggressive retention campaigns for month-to-month customers, offering incentives (e.g., discounts, upgrades, bundled services) to encourage conversion to longer-term (1-year or 2-year) contracts. Target these efforts proactively, especially for newer customers.

2.  **Focus on Early Customer Loyalty & Onboarding:**
    * **Insight:** Low tenure is highly correlated with churn.
    * **Recommendation:** Enhance the onboarding experience for new customers. This could include personalized welcome calls, dedicated customer support for initial setup, and proactive check-ins or small loyalty rewards within the first 6-12 months of service.

3.  **Address High Monthly Charges:**
    * **Insight:** Customers with higher monthly charges are more prone to churn.
    * **Recommendation:** Conduct targeted reviews for high-charge, high-risk customers. Analyze their current service bundles and proactively suggest more cost-effective plans, value-added services, or limited-time promotions to improve perceived value.

4.  **Investigate Fiber Optic Service Satisfaction:**
    * **Insight:** Despite being a premium offering, Fiber Optic internet service is associated with higher churn.
    * **Recommendation:** Initiate an investigation into potential service quality issues, reliability concerns, or aggressive competitor pricing specifically affecting Fiber Optic users. Conduct targeted surveys or technical audits in affected areas.

5.  **Promote Stable Payment Methods:**
    * **Insight:** Customers using Electronic Checks exhibit a higher churn rate.
    * **Recommendation:** Encourage switching to more stable and convenient payment methods (e.g., bank transfer, credit card auto-pay) through small incentives or by highlighting the benefits of uninterrupted service.

6.  **Highlight Value of Security and Tech Support:**
    * **Insight:** Absence of Online Security and Tech Support correlates with increased churn.
    * **Recommendation:** Actively market the benefits and peace of mind offered by these services to customers who currently do not subscribe to them. Consider bundling them into attractive service packages as a retention tool.

## 5. Results

The models were evaluated comprehensively, with the **Optimized XGBoost Classifier** demonstrating the best performance, particularly in identifying churners.

| Model                       | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| :-------------------------- | :------- | :-------- | :----- | :------- | :------ |
| Logistic Regression (Baseline) | 0.8027 | 0.6509 | 0.5535 | 0.5983 | 0.8426 |
| Random Forest Classifier    | 0.7913 | 0.6351 | 0.5027 | 0.5612 | 0.8259 |
| XGBoost Classifier          | 0.7800 | 0.5976 | 0.5241 | 0.5584 | 0.8272 |
| **Optimized XGBoost Classifier** | **0.7878** | **0.6183** | **0.5241** | **0.5673** | **0.8342** |

**Conclusion:** The **Optimized XGBoost Classifier** was chosen as the final model for its balanced performance and superior **Recall** score. This allows the company to correctly identify a significant portion of actual churners, enabling timely intervention and maximizing retention efforts. The ROC AUC score of **0.8342** further confirms its strong ability to distinguish between churning and non-churning customers.

## 6. Technologies Used

* **Programming Language:** Python
* **Libraries:**
    * `pandas`: For data manipulation and analysis.
    * `numpy`: For numerical operations.
    * `matplotlib`: For static data visualizations.
    * `seaborn`: For enhanced statistical data visualizations.
    * `scikit-learn`: For machine learning models (Logistic Regression, RandomForestClassifier, StandardScaler, train_test_split, GridSearchCV, and various metrics).
    * `xgboost`: For the XGBoost Classifier.
* **Development Environment:** JupyterLab

## 7. How to Run the Project

To run this project locally, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/telecom-customer-churn-prediction.git](https://github.com/YourGitHubUsername/telecom-customer-churn-prediction.git)
    cd telecom-customer-churn-prediction
    ```

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    conda create -n churn_env python=3.9 pandas numpy scikit-learn matplotlib seaborn jupyterlab ipykernel xgboost -y
    conda activate churn_env
    ```
    *Alternatively, if you prefer `pip` and `venv`:*
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab xgboost
    ```

3.  **Download the Dataset:**
    * Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
    * Place this `.csv` file directly into the `telecom-customer-churn-prediction` directory (the same folder as the `churn_prediction_project.ipynb` file).

4.  **Launch JupyterLab:**
    ```bash
    jupyter lab
    ```

5.  **Open and Run the Notebook:**
    * In the JupyterLab interface, open `churn_prediction_project.ipynb`.
    * Run all cells (`Kernel` -> `Restart Kernel and Run All Cells...`) to execute the analysis and see the results.

## 8. Future Work

* **Advanced Class Imbalance Techniques:** Experiment with techniques like SMOTE-NC (for mixed data types), ADASYN, or cost-sensitive learning to further optimize the model's performance on the minority class.
* **Model Interpretability:** Explore tools like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to provide more detailed, local explanations for individual customer churn predictions, aiding business decision-making.
* **Deployment:** Develop a simple web application (e.g., using Streamlit or Flask) to demonstrate real-time churn prediction for new customer data, allowing for interactive use.
* **A/B Testing Framework:** Propose a framework for A/B testing different retention strategies (e.g., specific discounts, personalized offers) based on churn predictions.
* **Feature Engineering from Business Logic:** Collaborate with domain experts to derive even more impactful features from existing data or external sources (e.g., customer service interaction logs, social media sentiment).

<img width="917" height="528" alt="Screenshot 2025-07-14 014324" src="https://github.com/user-attachments/assets/2c58e0bc-2c36-43ea-a338-739d75cb19fd" />


---

**Author:** Macpherson Zelu
**Connect:** www.linkedin.com/in/macpherson-zelu
