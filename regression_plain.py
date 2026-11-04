"""
Build a baseline regression model using the raw uncleaned data. This model will serve as a benchmark to compare the performance of more advanced models (cleaned data) that will be built later in the project. 
The baseline model will help us understand the initial performance and identify areas for improvement.
evaluete the model's performanc using r2 score as the main metric.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



#load the dataset data.csv
df=pd.read_csv("data.csv")

#Cleaning the data.
def clean_data_zscore(df):
    df = df.drop(columns=["Unnamed: 0"])  # Drop the index column if it exists 
    df = df.dropna(subset=["sale_price"])  # Drop rows with missing sale_price
    df = df.drop_duplicates()  # Drop duplicate rows
    # Using z-score to filter out extreme values for sale_price and mileage
    df['sale_price'] = np.log1p(df['sale_price'])  # Log-transform sale_price to reduce skewness
    # Z-score calculation for sale_price and mileage 
    z_score_sale_price = (df['sale_price'] - df['sale_price'].mean()) / df['sale_price'].std()
    z_score_mileage = (df['mileage'] - df['mileage'].mean()) / df['mileage'].std()
    df = df[(np.abs(z_score_sale_price) <= 3) & (np.abs(z_score_mileage) <= 3)].copy()  # Filter out extreme values based on z-score
    df["engine_cylinders"] = df["engine_cylinders"].str.extract(r"(\d+)", expand=False).astype(float)  # Extract numeric values from engine_cylinders. etc: 6_cylinders becomes just 6.
    return df

df = clean_data_zscore(df)
X = df.drop(columns=["sale_price"])  # Features
y = df["sale_price"]  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Linear regression model
model = Pipeline([("pre", pre), ("reg", LinearRegression())])
scores = cross_validate(model, X_train, y_train, cv=kf, scoring={'r2' : 'r2', 'neg_mean_squared_error': 'neg_mean_squared_error'})
print("Linear regression R² per fold:", scores["test_r2"])
print("Linear regression mean R²:", scores["test_r2"].mean(), "±", scores["test_r2"].std())
print("Linear regression MSE per fold:", scores["test_neg_mean_squared_error"])
print("Linear regression MSE:", scores["test_neg_mean_squared_error"].mean(), "±", scores["test_neg_mean_squared_error"].std())

# Visualization of the model's predictions vs true values
plt.figure(figsize=(8, 6))
y_pred = cross_val_predict(model, X_train, y_train, cv=kf)
plt.scatter(y_train, y_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Linear Regression Model on Training Data")
plt.savefig("Figures/Baselines_models/linearRegression_baseline.png")

# Random Forest Regressor model
rf_model = Pipeline([("pre", pre), ("reg", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10, max_samples=0.8, min_samples_leaf=5))]) # Added restirictions otherwise it takes too long to run
scores_rf = cross_validate(rf_model, X_train, y_train, cv=kf, scoring={'r2' : 'r2', 'neg_mean_squared_error': 'neg_mean_squared_error'})
print("Random Forest R² per fold:", scores_rf["test_r2"])
print("Random Forest mean R²:", scores_rf["test_r2"].mean(), "±", scores_rf["test_r2"].std())
print("Random Forest MSE per fold:", scores_rf["test_neg_mean_squared_error"])
print("Random Forest MSE:", scores_rf["test_neg_mean_squared_error"].mean(), "±", scores_rf["test_neg_mean_squared_error"].std())

# Visualization of the model's predictions vs true values
plt.figure(figsize=(8, 6))
y_pred_rf = cross_val_predict(rf_model, X_train, y_train, cv=kf)
plt.scatter(y_train, y_pred_rf, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Random Forest Model on Training Data")
plt.savefig("Figures/Baselines_models/randomForest_baseline.png")

# Ridge regression model
from sklearn.linear_model import Ridge
ridge_model = Pipeline([("pre", pre), ("reg", Ridge(alpha=1.0))])
scores_ridge = cross_validate(ridge_model, X_train, y_train, cv=kf, scoring={'r2' : 'r2', 'neg_mean_squared_error': 'neg_mean_squared_error'})
print("Ridge Regression R² per fold:", scores_ridge["test_r2"])
print("Ridge Regression mean R²:", scores_ridge["test_r2"].mean(), "±", scores_ridge["test_r2"].std())
print("Ridge Regression MSE per fold:", scores_ridge["test_neg_mean_squared_error"])
print("Ridge Regression MSE:", scores_ridge["test_neg_mean_squared_error"].mean(), "±", scores_ridge["test_neg_mean_squared_error"].std())

plt.figure(figsize=(8, 6))
y_pred_ridge = cross_val_predict(ridge_model, X_train, y_train, cv=kf)
plt.scatter(y_train, y_pred_ridge, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Ridge Regression Model on Training Data")
plt.savefig("Figures/Baselines_models/ridgeRegression_baseline.png")

# XGB Regressor model
xgb_model = Pipeline([("pre", pre), ("reg", XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10, subsample=0.8, min_child_weight=5))]) # Added restrictions otherwise it takes too long to run
scores_xgb = cross_validate(xgb_model, X_train, y_train, cv=kf, scoring={'r2' : 'r2', 'neg_mean_squared_error': 'neg_mean_squared_error'})
print("XGB Regressor R² per fold:", scores_xgb["test_r2"])
print("XGB Regressor mean R²:", scores_xgb["test_r2"].mean(), "±", scores_xgb["test_r2"].std())
print("XGB Regressor MSE per fold:", scores_xgb["test_neg_mean_squared_error"])
print("XGB Regressor MSE:", scores_xgb["test_neg_mean_squared_error"].mean(), "±", scores_xgb["test_neg_mean_squared_error"].std())

# Visualization of the model's predictions vs true values
plt.figure(figsize=(8, 6))
y_pred_xgb = cross_val_predict(xgb_model, X_train, y_train, cv=kf)
plt.scatter(y_train, y_pred_xgb, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("XGB Regressor Model on Training Data")
plt.savefig("Figures/Baselines_models/xgbRegression_baseline.png")


                                                                                    