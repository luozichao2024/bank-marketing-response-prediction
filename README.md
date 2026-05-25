# Bank Marketing Response Prediction

A machine learning project for predicting whether a bank customer will subscribe to a term deposit using the UCI Bank Marketing dataset.

This project includes data preprocessing, feature engineering, model training, hyperparameter tuning, threshold analysis, model evaluation, and feature importance analysis.

## Project Overview

Bank marketing campaigns often require identifying customers who are more likely to subscribe to financial products. This project builds a binary classification model to predict customer responses based on customer information, financial status, and marketing contact records.

The target variable is:

- `yes`: the customer subscribed to a term deposit
- `no`: the customer did not subscribe to a term deposit

## Dataset

The project uses the UCI Bank Marketing dataset.

Dataset link:  
https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset contains customer information such as:

- Age
- Job
- Marital status
- Education
- Account balance
- Housing loan
- Personal loan
- Contact type
- Campaign information
- Previous marketing outcome

## Project Structure

```text
bank-marketing-response-prediction/
├── README.md
├── requirements.txt
├── main.py
├── src/
│   ├── data_utils.py
│   ├── train_utils.py
│   └── visualize.py
└── results/
    ├── metrics.csv
    ├── classification_report.txt
    ├── feature_importance.csv
    ├── data_summary.txt
    └── figures/
        ├── target_distribution.png
        ├── confusion_matrix.png
        ├── roc_curve.png
        └── top_features.png
