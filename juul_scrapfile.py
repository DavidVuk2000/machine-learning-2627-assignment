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

size_text = 14   # change this one number to scale all text
plt.rcParams.update({
    'font.size':        size_text,        # base size (text, annotations)
    'axes.titlesize':   size_text + 2,    # subplot titles
    'axes.labelsize':   size_text,        # x and y axis labels
    'xtick.labelsize':  size_text - 2,    # tick numbers
    'ytick.labelsize':  size_text - 2,
    'legend.fontsize':  size_text - 2,
    'figure.titlesize': size_text - 2,    # fig.suptitle
})

"""
#load the dataset data.csv
df=pd.read_csv("data.csv")

# --- DATA EXPLORATION/DESCRIPTION ---
column_names = df.columns
print("Column names in the dataset: \n", column_names)
print("Shape of the dataset: \n", df.shape)

# Filtering the missing values
missing_values = df.isnull().sum()
print("Missing values in each column: \n" , missing_values)

# Placeholder strings in categorical columns
for col in cat_cols:
    print(col, df[col].unique()[:20])

# Suspicious numbers
print(df[num_cols].describe())   # look at min and max
print((df[num_cols] == 0).sum())

# NUMERICAL COLUMNS
# making correlation matrix with only the numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])
# removing the index row called 'Unnamed: 0' (this is the index column)
numeric_df = numeric_df.drop(columns=['Unnamed: 0'])
print("Numeric columns in the dataset: \n", numeric_df.columns)
print("Basic statistics of the numerical columns:\n", numeric_df.describe())
print("Double rows for numerical values: \n", numeric_df.duplicated().sum())

# CATIGORICAL COLUMNS
# Looking at the unique values in the categorical columns
classes_df = df.select_dtypes(include=['object'])
classes_df_20 = classes_df.loc[:, classes_df.nunique() <= 20] # Selecting all rows and only the columns with less than or equal to 20 unique values (for shape plot)

rows = []
for col in classes_df.columns:
    counts = df[col].value_counts()          # without NaN here, so the top value is a real category
    n_missing = df[col].isnull().sum()
    top_count = counts.iloc[0]
    rows.append({
        "feature": col,
        "n_unique": df[col].nunique(),
        "most_frequent": counts.index[0],
        "count": top_count,
        "percent": round(top_count / len(df) * 100, 2),
        "n_missing": n_missing,
        "percent_missing": round(n_missing / len(df) * 100, 2),
    })

cat_summary = pd.DataFrame(rows).sort_values("percent", ascending=False)
print(cat_summary.to_string(index=False))
print("Double rows for numerical values: \n", numeric_df.duplicated().sum())



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

df_clean = clean_data_zscore(df)
df_clean_numeric = df_clean.select_dtypes(include=['float64', 'int64'])
print("Basic statistics of the numerical columns:\n", df_clean_numeric.describe())

double_rows = df_clean_numeric.duplicated().sum()
print("Double values: \n", double_rows)

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
                     ("enc", TargetEncoder(random_state=42))]), target_cols), #I added a random_state to the TargetEncoder for reproducibility
])  # all 15 feature columns are used
"""

