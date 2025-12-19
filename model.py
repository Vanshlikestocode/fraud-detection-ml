import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv("fraud_data.csv")

# Encode categorical column
le = LabelEncoder()
df["TransactionType"] = le.fit_transform(df["TransactionType"])

# Features and target
X = df[["Amount", "TransactionType", "AccountBalance"]]
y = df["Fraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# User prediction
print("\n--- Fraud Detection ---")
amount = int(input("Enter transaction amount: "))
t_type = input("Enter transaction type (Debit/Credit): ")
balance = int(input("Enter account balance: "))

t_type_encoded = le.transform([t_type])[0]

input_data = [[amount, t_type_encoded, balance]]
result = model.predict(input_data)

if result[0] == 1:
    print("⚠️ Fraudulent Transaction Detected")
else:
    print("✅ Legitimate Transaction")
