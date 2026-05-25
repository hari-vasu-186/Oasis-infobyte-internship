import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("car data.csv")

print(data.head())

print(data.shape)

print(data.info())

print(data.describe())

print(data.isnull().sum())

sns.histplot(data['Selling_Price'], kde=True)
plt.show()

sns.countplot(x='Fuel_Type', data=data)
plt.show()

sns.countplot(x='Transmission', data=data)
plt.show()

encoder = LabelEncoder()

data['Fuel_Type'] = encoder.fit_transform(data['Fuel_Type'])
data['Selling_type'] = encoder.fit_transform(data['Selling_type'])
data['Transmission'] = encoder.fit_transform(data['Transmission'])

data['Car_Age'] = 2024 - data['Year']

data.drop(['Car_Name', 'Year'], axis=1, inplace=True)

X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

sample_data = pd.DataFrame(
    [[5.59, 27000, 1, 0, 1, 0, 6]],
    columns=X.columns
)

predicted_price = model.predict(sample_data)

print("Predicted Selling Price:",
      round(predicted_price[0], 2), "Lakhs")
