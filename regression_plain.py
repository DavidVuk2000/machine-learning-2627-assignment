"""
Build a baseline regression model using the raw uncleaned data. This model will serve as a benchmark to compare the performance of more advanced models (cleaned data) that will be built later in the project. 
The baseline model will help us understand the initial performance and identify areas for improvement.
evaluete the model's performanc using r2 score as the main metric.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import TargetEncoder




#load the dataset data.csv
df=pd.read_csv("data.csv")

#Cleaning the data.
def clean_data(df):
    df = df.drop(columns=["Unnamed: 0"])  # Drop the index column if it exists 
    df = df.dropna(subset=["sale_price"])  # Drop rows with missing sale_price
    df = df.drop_duplicates()  # Drop duplicate rows
    df = df[(df.sale_price > 50) & (df.sale_price < 200000)].copy()  # Filter out extreme sale prices
    df["engine_cylinders"] = df["engine_cylinders"].str.extract(r"(\d+)", expand=False).astype(float)  # Extract numeric values from engine_cylinders. etc: 6_cylinders becomes just 6.
    return df

df = clean_data(df)
X = df.drop(columns=["sale_price"])  # Features
y = np.log1p(df["sale_price"])  # Target variable (log-transformed sale price) instead of sale_price to reduce skewness and improve model performance


# Imputation + encoding.
num_cols = ["mileage", "year", "engine_cylinders"] #these are just the numerical columns that dont need encoding.
onehot_cols = ["brand", "fuel_type", "gearbox", "drive_type", "body_style", #here we have the categorical columns that will be one-hot encoded
               "color", "location", "title"]  # few distinct categories
target_cols = ["vehicle_condition", "vehicle_size", "vehicle_model", "region"]  # here we have the categorical columns that will be target encoded because of high cardinality.

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("ohe", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                      ("enc", OneHotEncoder(handle_unknown="ignore"))]), onehot_cols),
    ("te", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                     ("enc", TargetEncoder())]), target_cols),
])  # all 15 feature columns are used

model = Pipeline([("pre", pre), ("reg", LinearRegression())])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
print("R² per fold:", scores)
print("mean R²:", scores.mean(), "±", scores.std())
                                                                                    