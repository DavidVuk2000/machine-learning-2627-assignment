import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
warnings.filterwarnings("ignore")

target_options = ["sale_price", "body_style"]
numerical_columns = ["mileage", "year", "engine_cylinders", "sale_price"]
catagorical_columns_20 = ["fuel_type", "body_style", "vehicle_condition", "gearbox", "drive_type", "color", "vehicle_size", "title", "engine_cylinders_status"]
catagorical_columns_other = ["brand", "location", "vehicle_model", "region"]


# Im printing all column names

# Cleaning before splitting
def clean_raw(df, target):
    df = pd.DataFrame(df)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore").copy() # dropping unnamed: 0, the index column
    df = df.dropna(subset=[target]) 
    df = df.drop_duplicates() 

    # fixed rules for impossible values (applied to all rows, before the split)
    df = df[(df["sale_price"] >= 500) & (df["sale_price"] <= 300_000)]
    df = df[~(df["mileage"] > 500_000)]                                   # impossible mileage; ~ keeps NaN mileage


    cylinders = df["engine_cylinders"].str.extract(r"(\d+)", expand=False).astype(float) # converting the engine column
    df["engine_cylinders_status"] = np.where(cylinders.isna(), "Unknown", "Known")
    df["engine_cylinders"] = cylinders

    df["sale_price"] = np.log1p(df["sale_price"]) # log transforming the target value 
    return df

#Cleaning the numerical data
def clean_data_zscore(X, y, target):
    if target == "sale_price":
        z_price = (y - y.mean()) / y.std()
        z_mileage = (X["mileage"] - X["mileage"].mean()) / X["mileage"].std()

        out_price = z_price.abs() > 3
        out_mileage = z_mileage.abs() > 3
        remove = out_price | out_mileage

        return X[~remove], y[~remove]

    else:
        z_price = (X["sale_price"] - X["sale_price"].mean()) / X["sale_price"].std()
        z_mileage = (X["mileage"] - X["mileage"].mean()) / X["mileage"].std()

        out_price = z_price.abs() > 3
        out_mileage = z_mileage.abs() > 3

        remove = out_price | out_mileage

        return X[~remove], y[~remove]

def split_data(data, target):
    df = clean_raw(data, target)
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, y_train = clean_data_zscore(X_train, y_train, target)
    return X_train, X_test, y_train, y_test

def preprocessor(target):
    num_features = [col for col in numerical_columns if col != target]
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    transformers = [("num", num, num_features)]

    categorical_features = [col for col in catagorical_columns_20 if col != target]
    low = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False))])
    transformers.append(("low", low, categorical_features))

    cat_high_features = [col for col in catagorical_columns_other if col != target]
    high = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("target", TargetEncoder(target_type="auto", smooth="auto", cv=5, shuffle=True, random_state=42))])
    transformers.append(("high", high, cat_high_features))

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

