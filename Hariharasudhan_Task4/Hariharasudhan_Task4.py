import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("spam.csv", encoding='latin-1')

print(data.head())

print(data.shape)

print(data.info())


data = data[['v1', 'v2']]


data.columns = ['label', 'message']

print(data.head())

print(data.isnull().sum())


data['label'] = data['label'].map({'ham': 0, 'spam': 1})


sns.countplot(x=data['label'])
plt.show()


X = data['message']
y = data['label']


vectorizer = CountVectorizer()

X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = MultinomialNB()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

print(cm)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


sample = ["Congratulations! You won a free iPhone. Click now!"]

sample_data = vectorizer.transform(sample)

prediction = model.predict(sample_data)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam Message")
