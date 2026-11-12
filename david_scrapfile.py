import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

print("Plingg")

print("eow")

df = pd.read_csv("data.csv")

print(df["brand"].dtype)

df.info()

numeric_df = df.select_dtypes(include=['float64', 'int64'])

print("Numeric columns in the dataset: \n", numeric_df.columns)

for col in df.columns:
    if df[col].dtype == 'object': # checking if the column is categorical
        print(df[col].value_counts()) # with the default setting, .value_counts() does not show the count of NaN values (can be changed with dropna=False)
        print("---"*10)

#List of unique values in the vehicle_condition column
df["vehicle_condition"].unique()



for condition in df["vehicle_condition"].unique():
    condition_df = df[df["vehicle_condition"] == condition]
    plt.scatter(condition_df['vehicle_condition'], condition_df['sale_price'])

plt.xlabel('vehicle_condition')
plt.ylabel('sale_price')
plt.show()