import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("customer_churn.csv")

df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df = df.dropna()

le = LabelEncoder()

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = le.fit_transform(df[col])

print(df.dtypes)

x = df.drop("Churn", axis=1)
y = df["Churn"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()

model.fit(x_train, y_train)

print("Model Trained Successfully")

y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

importance = model.feature_importances_

feature_names = x.columns

feature_data = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_data = feature_data.sort_values(by="Importance", ascending=False)

print(feature_data)

plt.figure(figsize=(10,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_data
)

plt.title("Feature Importance")

plt.show()

import pickle

pickle.dump(model, open("churn_model.pkl", "wb"))

print("Model Saved Successfully")