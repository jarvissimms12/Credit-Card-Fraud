# Credit-Card-Fraud
Building a robust Credit Card Fraud Detection system to identify suspicious transactions in an imbalanced dataset. Using SMOTE for oversampling and Random Forest, the model achieves high recall for the minority class. Features include data scaling, feature engineering, and a saved pipeline with Joblib for real-world deployment.
Data Source: This project uses the Credit Card Fraud Detection dataset from Kaggle(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) . Please download the creditcard.csv file and place it in the /data folder to run the notebook.

## Project Overview
This project successfully implements a machine learning pipeline to identify fraudulent credit card transactions. By addressing the extreme class imbalance in the original dataset through SMOTE (Synthetic Minority Over-sampling Technique) and utilizing a Random Forest classification approach, the system achieves balanced performance in detecting both legitimate and fraudulent activity.

Data Preprocessing & Balancing
Feature Scaling: Input features were normalized using StandardScaler to ensure that varying scales of features (e.g., transaction amount vs. anonymized "V" components) did not bias the model.
Oversampling (SMOTE): To counter the "Needle in a Haystack" problem (where fraud is only ~0.17% of data), SMOTE was used to generate synthetic fraudulent examples. This balanced the training set, allowing the model to learn the distinct characteristics of fraud rather than simply guessing "legitimate" for every transaction.

## Features
- **Data Scaling:** Robust preprocessing using `StandardScaler`.
- **Imbalance Handling:** Implementation of SMOTE to improve model recall.
- **Model:** Trained a classifier (Decision Tree/Random Forest) achieving high precision and recall.
- **Persistence:** Model and Scaler are serialized using `joblib` for easy deployment.

## How to Run
1. Clone the repo: `git clone https://github.com/jarvissimms12/Credit-Card-Fraud`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook to see the evaluation metrics and confusion matrix.

## Results
After balancing the dataset with SMOTE and training the classifier, the model was evaluated on a test set. The results demonstrate high reliability in detecting fraudulent transactions.
Classification Report - Overall Accuracy: 90% 
Confusion Matrix Insights
As shown in the confusion matrix from the notebook:True Positives: 128 frauds correctly identified. False Negatives: Only 14 frauds were missed. False Positives: 13 legitimate transactions were flagged as fraud. 
Feature ImportanceThe model identified key anonymized features (V-components) as the primary indicators of fraudulent behavior.

The project is built for real-world application with the following serialized components:
credit_card_fraud_model.pkl: The trained intelligence of the system.
scaler.pkl: The mathematical parameters used to normalize new incoming transaction data.
