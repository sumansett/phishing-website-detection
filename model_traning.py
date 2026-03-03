import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score

import joblib



# Load dataset

df = pd.read_csv("Dataset\\website_analysis.csv")
print("Shape:", df.shape)



# Split X and y

X = df.drop("result", axis=1)
y = df["result"]



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scaling (needed for LR + SVM)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Logistic Regression

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred, pos_label=1)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}, F1 Score: {lr_f1:.4f}")


# SVM

svm_model = SVC(kernel="rbf", probability=True)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, pos_label=1)
print(f"SVM Accuracy: {svm_acc:.4f}, F1 Score: {svm_f1:.4f}")



# Random Forest

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, pos_label=1)
print(f"Random Forest Accuracy: {rf_acc:.4f}, F1 Score: {rf_f1:.4f}")



# Compare and select best model
scores = {
    "Logistic Regression": lr_f1,
    "SVM": svm_f1,
    "Random Forest": rf_f1
}


best_model_name = max(scores, key=scores.get)

if best_model_name == "Logistic Regression":
    best_model = lr_model
elif best_model_name == "SVM":
    best_model = svm_model
else:
    best_model = rf_model



print("\nBest Model:", best_model_name)


# Save best model + scaler + feature names

joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("\nModel, scaler, and feature names saved successfully.")

