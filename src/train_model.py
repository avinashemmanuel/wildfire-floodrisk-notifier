import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. LOAD DATA
df = pd.read_csv("data/processed/firms_dataset.csv")

# 2. FEATURES AND TARGET
X = df[[
    "bright_ti4",
    "bright_ti5",
    "frp",
    "confidence",
    "daynight"
]]

y = df["risk"]

# 3.TRAIN, TEST, SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. MODEL
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# 5. EVALUATION
y_prob = model.predict_proba(X_test)[:, 1]

threshold = 0.4
y_pred = (y_prob > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy, 3))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 6. SAVE MODEL
joblib.dump(model, "model.pkl")