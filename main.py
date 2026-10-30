print("HEYO~!")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")

# --- DATA EXPLORATION ---
# Filtering the missing values
missing_values = df.isnull().sum()
print("Missing values in each column: \n" , missing_values)


# DIRTY DATA
# --- DATA VISUALIZATION ---
#...

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