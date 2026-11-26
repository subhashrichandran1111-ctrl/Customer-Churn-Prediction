
# Customer Churn Prediction

## Project Overview
This project predicts whether a telecom customer is likely to leave the company using Machine Learning.

The project uses the IBM Telco Customer Churn dataset.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Machine Learning Models
1. Logistic Regression
2. Random Forest Classifier
3. Balanced Logistic Regression
4. Tuned Logistic Regression

## Best Model
Tuned Logistic Regression

Test Results:
- Accuracy: 74.24%
- Precision: 50.96%
- Recall: 78.34%
- F1-Score: 61.75%
- ROC-AUC: 84.14%

## Project Structure

customer-churn-prediction/
- data/
- notebooks/
- models/
- src/
- app.py
- requirements.txt
- README.md

## Dataset
IBM Telco Customer Churn Dataset:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download the CSV and place it inside the data folder.

## How to Run the Project

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run the Streamlit application

streamlit run app.py

### 3. Open the application

http://localhost:8501

## Project Workflow
1. Dataset collection
2. Exploratory Data Analysis
3. Data preprocessing
4. Model development
5. Model evaluation
6. Model improvement
7. Hyperparameter tuning
8. Model saving
9. Streamlit deployment

## Conclusion
The tuned Logistic Regression model achieved a recall of 78.34% and an F1-score of 61.75% on the test dataset.

The application allows users to enter customer details and receive a churn prediction.

## Author

**Subhashri C**

