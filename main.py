print("HEYO~!")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")

# Filtering the missing values
missing_values = df.isnull().sum()
print("Missing values in each column: \n" , missing_values)

print(df['title'])