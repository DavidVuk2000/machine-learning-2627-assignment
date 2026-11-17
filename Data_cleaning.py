import warnings
import math
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

warnings.filterwarnings("ignore")

#load the dataset data.csv
df=pd.read_csv("data.csv")
# Im printing all column names
print(df.columns)


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

n = len(numeric_df_after.columns)
ncols = 2
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
axes = np.atleast_1d(axes).flatten()      # must come BEFORE the loop

for i, column in enumerate(numeric_df_after.columns):
    ax = axes[i]
    data = numeric_df_after[column].dropna()

    if column == 'sale_price':
        data = data[data > 0]                       # log scale can't show 0
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 31)
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xscale('log')
        ax.set_xlabel(f"{column} (log scale)")
    else:
        ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(column)

    ax.set_title(f"Histogram of {column}")
    ax.set_ylabel("Frequency")

# hide unused cells
for j in range(n, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("Figures/DataVisualization/histograms_numerical_after.png", dpi=300)
plt.show()

