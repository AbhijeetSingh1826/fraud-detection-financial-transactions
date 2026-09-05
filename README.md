# Real-Time Fraud Detection in Financial Transactions

A machine learning system that detects fraudulent credit card transactions in real time, built end-to-end from raw data to a deployed live dashboard.

## Live Demo
🔗 [View Dashboard](https://fraud-detection-financial-transactions-vftwhcjbgjywtomvqtrr7s.streamlit.app/)

## Problem
Credit card fraud accounts for a tiny fraction of transactions (~0.17% in this dataset) but causes significant financial loss. Standard accuracy metrics are misleading on such imbalanced data, so this project focuses on precision/recall tradeoffs and business-relevant evaluation.

## Approach
- **EDA**: Analyzed class imbalance and transaction amount patterns across fraud vs. legitimate transactions
- **Preprocessing**: Stratified train/test split, scaled Amount/Time features, applied SMOTE to the training set only (never the test set, to avoid data leakage)
- **Modeling**: Compared Logistic Regression, Random Forest, and XGBoost using Precision, Recall, F1, ROC-AUC, and PR-AUC (chosen over plain accuracy due to class imbalance)
- **Deployment**: Wrapped the winning model in a FastAPI endpoint for real-time scoring, simulated a live transaction stream, and built a Streamlit dashboard for monitoring

## Results
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (SMOTE) | 0.058 | 0.918 | 0.109 | 0.970 |
| Random Forest | 0.428 | 0.878 | 0.575 | 0.984 |
| XGBoost (final model) | 0.607 | 0.867 | 0.714 | 0.977 |

XGBoost was selected as the final model for its best balance of precision and recall (F1 = 0.714), minimizing false alarms while still catching 86.7% of fraudulent transactions.

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, imbalanced-learn (SMOTE), FastAPI, Streamlit, Plotly

## Project Structure
data/ - dataset (raw data gitignored)
notebooks/ - EDA and model development
src/ - API and stream simulation scripts
models/ - saved trained models
app/ - Streamlit dashboard


## How to Run Locally
```bash
pip install -r requirements.txt
python -m uvicorn src.api:app --reload
python -m streamlit run app/dashboard.py
```