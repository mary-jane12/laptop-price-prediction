import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    df = pd.read_csv("data/laptop_data.csv")
except FileNotFoundError:
    df = pd.read_csv("laptop_data.csv")

print("Dataset Shape:", df.shape)
print(df.info())
print(df.describe())

# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Display first 5 rows
print(df.head())
