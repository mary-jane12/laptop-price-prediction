import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load
try:
    df = pd.read_csv("data/laptop_data.csv")
except FileNotFoundError:
    df = pd.read_csv("laptop_data.csv")

# Example columns; adjust to your CSV’s actual names
target = "Price"               # sometimes it's 'Price_euros' etc.
numeric = [c for c in df.columns if df[c].dtype != 'object' and c != target]
categorical = [c for c in df.columns if df[c].dtype == 'object']

df = df.dropna(subset=[target])  # keep rows with price
X = df.drop(columns=[target])
y = df[target]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

pipe = Pipeline([
    ("prep", pre),
    ("model", LinearRegression())
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)
print("R2:", round(r2_score(yte, pred), 3), "MAE:", round(mean_absolute_error(yte, pred), 2))
