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

#Cleaning the data.
def clean_data(df):
    df = df.drop(columns=["Unnamed: 0"])  # Drop the index column if it exists 
    df = df.dropna(subset=["sale_price"])  # Drop rows with missing sale_price
    df = df.drop_duplicates()  # Drop duplicate rows
    df = df[(df.sale_price > 50) & (df.sale_price < 200000)].copy()  # Filter out extreme sale prices
    df = df[(df.mileage > 0) & (df.mileage < 300000)].copy()  # Filter out extreme mileage values
    df["engine_cylinders"] = df["engine_cylinders"].str.extract(r"(\d+)", expand=False).astype(float)  # Extract numeric values from engine_cylinders. etc: 6_cylinders becomes just 6.
    return df

df = clean_data(df)
X = df.drop(columns=["sale_price"])  # Features
y = np.log1p(df["sale_price"])  # Target variable (log-transformed sale price) instead of sale_price to reduce skewness and improve model performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set size:", X_train.shape)

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

# Define the KFold cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# XGB Regressor model
xgb = Pipeline([("pre", pre), ("reg", XGBRegressor(n_estimators=100, random_state=42))]) 

# Tweaking the XGB model to see if we can improve the performance by changing hyperparameters using RandomizedSearchCV
# Defining dictionary of hyperparameters to search
param_distributions = {
    'reg__n_estimators': [50, 100, 200],
    'reg__max_depth': [None, 10],
    'reg__learning_rate': [0.01, 0.1, 0.2],
}

# Setting up the RandomizedSearchCV
xgb_random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_distributions, n_iter=20, cv=kf, verbose=2, random_state=42, n_jobs=-1)
xgb_random_search.fit(X_train, y_train)
# Getting the best parameters from the random search
best_params = xgb_random_search.best_params_
print("Best parameters from RandomizedSearchCV:", best_params)
best_xgb_model = xgb_random_search.best_estimator_

# Using the best parameters and best estimator to fit a new XGB model
best_xgb_model.fit(X_train, y_train)

# Making predictions with the best XGB model
y_pred_best_xgb_test = best_xgb_model.predict(X_test)
#performance metrics for the best XGB model
rmse_best_xgb = np.sqrt(mean_squared_error(y_test, y_pred_best_xgb_test))
mae_best_xgb = mean_absolute_error(y_test, y_pred_best_xgb_test)
r2_best_xgb = r2_score(y_test, y_pred_best_xgb_test)
print(f"Best XGB - Root Mean Squared Error (RMSE): {rmse_best_xgb}")
print(f"Best XGB - Mean Absolute Error (MAE): {mae_best_xgb}")
print(f"Best XGB - R^2 Score: {r2_best_xgb}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best_xgb_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
# Adding the best parameters as text to the plot
plt.text(0.05, 0.95, f"Best Parameters:\n{best_params}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.title("Best XGB Model on Test Data")
plt.savefig("Figures/best_xgb.png")
plt.show()