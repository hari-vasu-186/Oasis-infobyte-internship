
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Unemployment in India.csv")

print("FIRST 5 ROWS OF DATASET")
print(df.head())
print("\nDATASET INFORMATION")
print(df.info())
print("\nMISSING VALUES")
print(df.isnull().sum())

df.columns = [
    "States",
    "Date",
    "Frequency",
    "Estimated Unemployment Rate",
    "Estimated Employed",
    "Estimated Labour Participation Rate",
    "Region"
]

df["Date"] = pd.to_datetime(df["Date"],dayfirst=True)

df = df.dropna()


sns.set_style("whitegrid")


plt.figure(figsize=(10,6))
sns.histplot(
    df["Estimated Unemployment Rate"],
    bins=20,
    kde=True
)

plt.title("Distribution of Unemployment Rate")
plt.xlabel("Unemployment Rate")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(12,6))

sns.barplot(
    x="Region",
    y="Estimated Unemployment Rate",
    data=df
)

plt.title("Average Unemployment Rate by Region")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))

numeric_df = df.select_dtypes(include=['float64', 'int64'])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(15,7))

sns.barplot(
    x="States",
    y="Estimated Unemployment Rate",
    data=df
)

plt.xticks(rotation=90)
plt.title("State Wise Unemployment Rate")
plt.show()



avg_rate = df["Estimated Unemployment Rate"].mean()

print("\nAVERAGE UNEMPLOYMENT RATE")
print(avg_rate)

highest_states = df.groupby("States")[
    "Estimated Unemployment Rate"
].mean().sort_values(ascending=False)

print("\nTOP 10 STATES WITH HIGHEST UNEMPLOYMENT")
print(highest_states.head(10))

print("\nPROJECT COMPLETED SUCCESSFULLY")
