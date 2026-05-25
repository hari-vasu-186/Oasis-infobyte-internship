# ==========================================
# UNEMPLOYMENT ANALYSIS WITH PYTHON
# AICTE OASIS INFOBYTE INTERNSHIP PROJECT
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Unemployment in India.csv")

# Display First Rows
print("FIRST 5 ROWS OF DATASET")
print(df.head())

# Dataset Information
print("\nDATASET INFORMATION")
print(df.info())

# Check Missing Values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Rename Columns
df.columns = [
    "States",
    "Date",
    "Frequency",
    "Estimated Unemployment Rate",
    "Estimated Employed",
    "Estimated Labour Participation Rate",
    "Region"
]

# Convert Date Column
df["Date"] = pd.to_datetime(df["Date"],dayfirst=True)

# Remove Missing Values
df = df.dropna()

# ==========================================
# DATA VISUALIZATION
# ==========================================

sns.set_style("whitegrid")

# 1. Distribution Plot
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

# 2. Region Wise Unemployment
plt.figure(figsize=(12,6))

sns.barplot(
    x="Region",
    y="Estimated Unemployment Rate",
    data=df
)

plt.title("Average Unemployment Rate by Region")
plt.xticks(rotation=45)
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(8,5))

numeric_df = df.select_dtypes(include=['float64', 'int64'])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# 4. State Wise Analysis
plt.figure(figsize=(15,7))

sns.barplot(
    x="States",
    y="Estimated Unemployment Rate",
    data=df
)

plt.xticks(rotation=90)
plt.title("State Wise Unemployment Rate")
plt.show()

# ==========================================
# ANALYSIS
# ==========================================

# Average Unemployment Rate
avg_rate = df["Estimated Unemployment Rate"].mean()

print("\nAVERAGE UNEMPLOYMENT RATE")
print(avg_rate)

# Top 10 Highest States
highest_states = df.groupby("States")[
    "Estimated Unemployment Rate"
].mean().sort_values(ascending=False)

print("\nTOP 10 STATES WITH HIGHEST UNEMPLOYMENT")
print(highest_states.head(10))

print("\nPROJECT COMPLETED SUCCESSFULLY")
