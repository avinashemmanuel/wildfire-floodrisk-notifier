import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

#load data
df = pd.read_csv("data/processed/real_dataset.csv")

X = df.drop("risk", axis=1)
y = df["risk"]

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#train
model = XGBClassifier()
model.fit(X_train, y_train)

#evaluate
accuracy = model.score(X_test, y_test)
print("Accuracy: ", accuracy)

#save model
joblib.dump(model, "model.pkl")

print("Model saved!")