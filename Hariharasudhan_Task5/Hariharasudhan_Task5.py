import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("Advertising.csv")

print(data.head())

print(data.shape)

print(data.info())

print(data.describe())

print(data.isnull().sum())

# Remove unnecessary column if present
if 'Unnamed: 0' in data.columns:
    data.drop('Unnamed: 0', axis=1, inplace=True)

# Correlation Heatmap
sns.heatmap(data.corr(), annot=True)
plt.show()

# TV vs Sales
plt.scatter(data['TV'], data['Sales'])
plt.xlabel("TV Advertisement")
plt.ylabel("Sales")
plt.show()

# Features and target
X = data.drop('Sales', axis=1)
y = data['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.show()

# Predict new sales
sample = pd.DataFrame(
    [[230.1, 37.8, 69.2]],
    columns=X.columns
)

prediction = model.predict(sample)

print("Predicted Sales:", prediction[0])
