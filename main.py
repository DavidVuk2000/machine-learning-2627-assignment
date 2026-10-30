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

# ++++ DIRTY DATA ++++
# --- DATA VISUALIZATION ---
# NUMERICAL COLUMNS
# making correlation matrix with only the numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])
# removing the index row called 'Unnamed: 0' (this is the index column)
numeric_df = numeric_df.drop(columns=['Unnamed: 0'])
print("Numeric columns in the dataset: \n", numeric_df.columns)
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("Figures/correlation_matrix_numerical.png")

# CATIGORICAL COLUMNS
# Looking at the unique values in the categorical columns
classes_df = df.select_dtypes(include=['object'])
classes_df_20 = classes_df.loc[:, classes_df.nunique() <= 20] # Selecting all rows and only the columns with less than or equal to 20 unique values (for shape plot)

number_of_unique_values = classes_df.nunique()
print("Number of unique values in each categorical column: \n", number_of_unique_values)

# Visualizing the count of unique values in the categorical columns (<= 20 unique values)
fig, axes = plt.subplots(nrows=len(classes_df_20.columns), ncols=1, figsize=(10, 10 * len(classes_df_20.columns)))
for column in classes_df.columns:
    if classes_df[column].nunique() > 20:
        print(f"Column {column} has more than 20 unique values, skipping count plot.")
        continue
    sns.countplot(data=df, x=column, ax=axes[classes_df_20.columns.get_loc(column)])
    axes[classes_df_20.columns.get_loc(column)].set_title(f"Count Plot for {column}")
    axes[classes_df_20.columns.get_loc(column)].tick_params(axis='x', labelrotation=45)
plt.savefig(f"Figures/count_plot.png")

# Looking at correlations for the catigorical columns (<= 10 unique values)
classes_df_10 = classes_df.loc[:, classes_df.nunique() <= 10] # Selecting all rows and only the columns with less than or equal to 5 unique values (for correlation plot)
# Looking at the binary variables (0/1) in the categorical columns (<= 20 unique values)
print(classes_df.columns)
categorical_df = pd.get_dummies(classes_df_10)
print("Categorical columns after one-hot encoding: \n", categorical_df.columns)

# Negative corellations (most of the negative correlations are negative because they orignate from the same)!!
# Making correlation heatmap for the categorical columns (<= 10 unique values)
correlation_matrix_categorical = categorical_df.corr()
plt.figure(figsize=(25, 25))
sns.heatmap(correlation_matrix_categorical, annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1)
plt.title("Correlation Matrix for Categorical Columns (<= 10 unique values)")
plt.savefig("Figures/correlation_matrix_categorical.png")

# NUMERICAL AND CATIGORICAL COLUMNS
# Combining this with numerical columns
num_cat_df = pd.concat([categorical_df, numeric_df], axis=1)
# Making correlation heatmap for the categorical columns (<= 10 unique values) and numerical columns
correlation_matrix_num_cat = num_cat_df.corr()
plt.figure(figsize=(25, 25))
sns.heatmap(correlation_matrix_num_cat, annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1)
plt.title("Correlation Matrix for Categorical Columns (<= 10 unique values) & numerical values")
plt.savefig("Figures/correlation_matrix_num_cat.png")

# --- DEFINE MODELS ---
# --- regression model ---
#...

# --- classification model ---
#...

# --- DATA CLEANING ---
#...

# ++++ CLEAN DATA ++++
# --- DATA VISUALIZATION ---
#...

# --- DEFINE MODELS ---
# --- regression model ---
#...

# --- classification model ---
#...