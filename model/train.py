import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

CSV_PATH = "model/data/diabetes.csv"
print(f"Loading data from {CSV_PATH}...")

df = pd.read_csv(CSV_PATH, header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Dataset shape: {X.shape}, classes: {set(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {acc:.4f}")

os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to model/model.pkl")
