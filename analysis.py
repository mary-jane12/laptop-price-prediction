import pandas as pd
try:
    df = pd.read_csv("data/laptop_data.csv")
except FileNotFoundError:
    df = pd.read_csv("laptop_data.csv")
print("Shape:", df.shape)
print(df.head(10))
