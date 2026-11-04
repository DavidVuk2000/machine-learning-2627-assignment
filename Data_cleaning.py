import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import time

warnings.filterwarnings("ignore")

#load the dataset data.csv
df=pd.read_csv("data.csv")

# Histograms of the numerical values before cleaning with z-score
numeric_df_before = df.select_dtypes(include=['float64', 'int64'])
numeric_df_before = numeric_df_before.drop(columns=['Unnamed: 0'])
fig, axes = plt.subplots(nrows=len(numeric_df_before.columns), ncols=1, figsize=(10, 5 * len(numeric_df_before.columns)))
for column in numeric_df_before.columns:
    axes[numeric_df_before.columns.get_loc(column)].hist(numeric_df_before[column], bins=30, edgecolor='black', alpha=0.7)
    axes[numeric_df_before.columns.get_loc(column)].set_title(f"Histogram of {column}")
    axes[numeric_df_before.columns.get_loc(column)].set_xlabel(column)
    axes[numeric_df_before.columns.get_loc(column)].set_ylabel("Frequency")
plt.savefig("Figures/DataVisualization/histograms_numerical_before.png")

#Cleaning the data.
def clean_data_zscore(df):
    df = df.drop(columns=["Unnamed: 0"])  # Drop the index column if it exists 
    df = df.dropna(subset=["sale_price"])  # Drop rows with missing sale_price
    df = df.drop_duplicates()  # Drop duplicate rows
    # Using z-score to filter out extreme values for sale_price and mileage
    df['sale_price'] = np.log1p(df['sale_price'])  # Log-transform sale_price to reduce skewness
    z_score_sale_price = (df['sale_price'] - df['sale_price'].mean()) / df['sale_price'].std()
    z_score_mileage = (df['mileage'] - df['mileage'].mean()) / df['mileage'].std()
    df = df[(np.abs(z_score_sale_price) <= 3) & (np.abs(z_score_mileage) <= 3)].copy()  # Filter out extreme values based on z-score
    df["engine_cylinders"] = df["engine_cylinders"].str.extract(r"(\d+)", expand=False).astype(float)  # Extract numeric values from engine_cylinders. etc: 6_cylinders becomes just 6.
    return df

df = clean_data_zscore(df)

# Histograms of the numerical values after cleaning with z-score
numeric_df_after = df.select_dtypes(include=['float64', 'int64'])
fig, axes = plt.subplots(nrows=len(numeric_df_after.columns), ncols=1, figsize=(10, 5 * len(numeric_df_after.columns)))
for column in numeric_df_after.columns:
    axes[numeric_df_after.columns.get_loc(column)].hist(numeric_df_after[column], bins=30, edgecolor='black', alpha=0.7)
    axes[numeric_df_after.columns.get_loc(column)].set_title(f"Histogram of {column}")
    axes[numeric_df_after.columns.get_loc(column)].set_xlabel(column)
    axes[numeric_df_after.columns.get_loc(column)].set_ylabel("Frequency")
plt.savefig("Figures/DataVisualization/histograms_numerical_after.png")

