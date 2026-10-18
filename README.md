# Fraud Detection (ML)

**Flags potentially fraudulent transactions with logistic regression.** A small, readable project: one dataset, one model, an evaluation, and a prompt that scores a transaction you type in.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

## What it does

1. Loads `fraud_data.csv`, which has the columns `Amount`, `TransactionType` (Debit or Credit), `AccountBalance` and `Fraud` (0 or 1).
2. Encodes the transaction type as a number.
3. Trains a logistic-regression model on 70% of the rows and tests it on the other 30%.
4. Prints the accuracy and a full classification report (precision, recall, F1).
5. Asks you for an amount, a type and a balance, and says whether that transaction looks fraudulent.

## Run it

```bash
pip install -r requirements.txt
python model.py
```

## Next steps

This is a learning project on a small dataset. The bigger version is [SMG Fraud Detection](https://github.com/Vanshlikestocode/SMG-Fraud-Detection): a model trained on a public credit-card dataset and served behind a FastAPI endpoint with its own web interface.
