
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



data = pd.read_csv("Iris.csv")

print("========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== DATASET SHAPE ==========")
print(data.shape)

print("\n========== DATASET INFO ==========")
print(data.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(data.describe())




if 'Id' in data.columns:
    data.drop('Id', axis=1, inplace=True)


print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())




sns.pairplot(data, hue='Species')
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()


plt.figure(figsize=(8,6))

sns.heatmap(
    data.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()



X = data.drop('Species', axis=1)
y = data['Species']


encoder = LabelEncoder()
y = encoder.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)



model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\n========== MODEL TRAINED SUCCESSFULLY ==========")



y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", accuracy)


print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)


plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()



sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(sample)

predicted_species = encoder.inverse_transform(prediction)

print("\n========== NEW FLOWER PREDICTION ==========")
print("Predicted Species:", predicted_species[0])



print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========")
