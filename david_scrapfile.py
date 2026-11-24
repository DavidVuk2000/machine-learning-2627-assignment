import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["toolbar"] = "None"

#%%

#Cleaning the data.

def clean_data_David_v1(df):
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


df = pd.read_csv("data.csv")

df.head()

print("1")
 
#%%

print("2")
numeric_df = df.select_dtypes(include=['float64', 'int64'])

print("wa:" + str(df['vehicle_condition'].dtype))
for col in df.columns:
    if df[col].dtype == 'str': # checking if the column is categorical
        print(df[col].value_counts()) # with the default setting, .value_counts() does not show the count of NaN values (can be changed with dropna=False)
        print("---"*10)


#List of unique values in the vehicle_condition column
df["vehicle_condition"].unique()

cleaned_df = clean_data_David_v1(df)

conditions = cleaned_df["vehicle_condition"].dropna().unique()
bins = np.histogram_bin_edges(cleaned_df["sale_price"], bins=30)
fig, axes = plt.subplots(6, 1, sharex=True, figsize=(5, 10))

for ax, condition in zip(axes, conditions):
    condition_prices = cleaned_df.loc[
        cleaned_df["vehicle_condition"] == condition, "sale_price"
    ]
    ax.hist(condition_prices, bins=bins)
    ax.set_title(f"Condition {condition}")
    ax.set_xlabel("sale_price (log1p)")

axes[0].set_ylabel("Count")
fig.tight_layout(h_pad=3.0)
plt.show()

#%%

body_styles = cleaned_df["body_style"].dropna().unique()
fig, axes = plt.subplots(
    len(body_styles), 1, sharex=True,
    figsize=(5, max(10, 2.5 * len(body_styles))),
    squeeze=False,
)

for ax, body_style in zip(axes[:, 0], body_styles):
    body_style_prices = cleaned_df.loc[
        cleaned_df["body_style"] == body_style, "sale_price"
    ]
    ax.hist(body_style_prices, bins=bins)
    ax.set_title(f"Body style: {body_style}")
    ax.set_xlabel("sale_price (log1p)")

axes[0, 0].set_ylabel("Count")
fig.tight_layout(h_pad=3.0)
plt.show()