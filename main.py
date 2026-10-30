print("HEYO~!")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")

# --- DATA EXPLORATION ---
column_names = df.columns
print("Column names in the dataset: \n", column_names)
print("Shape of the dataset: \n", df.shape)

# Filtering the missing values
missing_values = df.isnull().sum()
print("Missing values in each column: \n" , missing_values)

# DIRTY DATA
# --- DATA VISUALIZATION ---
# making correlation matrix with only the numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])
# removing the index row called 'Unnamed: 0' (this is the index column)
numeric_df = numeric_df.drop(columns=['Unnamed: 0'])
print("Numeric columns in the dataset: \n", numeric_df.columns)
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("Figures/correlation_matrix.png")


# --- DEFINE MODELS ---
# --- regression model ---
#...

# --- classification model ---
#...

# --- DATA CLEANING ---
#...

# CLEAN DATA
# --- DATA VISUALIZATION ---
#...

# --- DEFINE MODELS ---
# --- regression model ---
#...

# --- classification model ---
#...