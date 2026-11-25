# Regression: predict the car price (log1p of sale_price)
# Models: Dummy (floor), Ridge (linear baseline), Random Forest, XGBoost
# Every model: 5-fold CV on the training set + one evaluation on the untouched test set.
# tuning is done with GridSearchCV (Ridge) or RandomizedSearchCV (RF, XGB). 


import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from Data_cleaning import split_data, preprocessor


# ======================================================================
# 1. Setup
# ======================================================================
TARGET = "sale_price"
RANDOM_STATE = 42
N_ITER_RF = 20      # random-search combinations for Random Forest (use 3 for a quick test run)
N_ITER_XGB = 30     # random-search combinations for XGBoost     (use 3 for a quick test run)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

raw = pd.read_csv("data.csv")
X_train, X_test, y_train, y_test = split_data(raw, TARGET)   # y is already log1p(price)
print("Train:", X_train.shape, "| Test:", X_test.shape)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)   # same folds for every model

# metrics computed in every CV fold (sklearn reports errors as negative numbers)
SCORING = {"r2": "r2", "mse": "neg_mean_squared_error", "mae": "neg_mean_absolute_error"}

cv_rows = []        # one row per model for the CV table
test_rows = []      # one row per model for the test table
best_params = {}    # best hyperparameters of the tuned models
predictions = pd.DataFrame({"y_true_log": y_test.values,
                            "y_true_$": np.expm1(y_test.values)})


# ======================================================================
# Helper functions (same calculation for every model)
# ======================================================================
def cv_summary(name, cv_result):
    """Mean CV scores from cross_validate(); errors turned back into positive numbers."""
    return {"model": name,
            "CV_R2_log": cv_result["test_r2"].mean(),
            "CV_R2_std": cv_result["test_r2"].std(),
            "train_R2_log": cv_result["train_r2"].mean(),          # for the overfitting check
            "CV_MSE_log": -cv_result["test_mse"].mean(),
            "CV_MAE_log": -cv_result["test_mae"].mean()}


def search_summary(name, search):
    """Same numbers as cv_summary, but taken from the best combination of a grid/random search."""
    res, i = search.cv_results_, search.best_index_
    return {"model": name,
            "CV_R2_log": res["mean_test_r2"][i],
            "CV_R2_std": res["std_test_r2"][i],
            "train_R2_log": res["mean_train_r2"][i],
            "CV_MSE_log": -res["mean_test_mse"][i],
            "CV_MAE_log": -res["mean_test_mae"][i]}


def test_summary(name, model):
    """Predict the test set once; metrics on the log scale and in dollars."""
    pred_log = model.predict(X_test)
    true_price, pred_price = np.expm1(y_test), np.expm1(pred_log)  # back from log1p to dollars
    predictions[f"{name}_pred_log"] = pred_log
    predictions[f"{name}_pred_$"] = pred_price
    return {"model": name,
            "test_R2_log": r2_score(y_test, pred_log),
            "test_MSE_log": mean_squared_error(y_test, pred_log),
            "test_RMSE_$": np.sqrt(mean_squared_error(true_price, pred_price)),
            "test_MAE_$": mean_absolute_error(true_price, pred_price)}


def clean_params(params):
    """Remove the 'model__' prefix that the Pipeline adds to parameter names."""
    return {k.replace("model__", ""): v for k, v in params.items()}


# ======================================================================
# 2. Dummy regressor (always predicts the median) - the baseline model. we dont tune anything here.
# ======================================================================
print("\n--- Dummy ---")
dummy = Pipeline([("prep", preprocessor(TARGET)),
                  ("model", DummyRegressor(strategy="median"))])

dummy_cv = cross_validate(dummy, X_train, y_train, cv=cv, scoring=SCORING, return_train_score=True)
cv_rows.append(cv_summary("Dummy", dummy_cv))

dummy.fit(X_train, y_train)                         # refit on the full training set
test_rows.append(test_summary("Dummy", dummy))
joblib.dump(dummy, "models/dummy.joblib")
print(cv_rows[-1], "\n", test_rows[-1])


# ======================================================================
# 3. Ridge regression - linear baseline (only alpha is tuned here)
# ======================================================================
print("\n--- Ridge ---")
ridge = Pipeline([("prep", preprocessor(TARGET)),
                  ("model", Ridge())])

ridge_grid = {"model__alpha": [0.1, 1, 10, 100]}    # regularisation strength
ridge_search = GridSearchCV(ridge, ridge_grid, cv=cv, scoring=SCORING, refit="r2",
                            return_train_score=True, n_jobs=-1)
ridge_search.fit(X_train, y_train)                  # also refits the best alpha on the full training set

cv_rows.append(search_summary("Ridge", ridge_search))
best_params["Ridge"] = clean_params(ridge_search.best_params_)
test_rows.append(test_summary("Ridge", ridge_search.best_estimator_))
joblib.dump(ridge_search.best_estimator_, "models/ridge.joblib")
print("best:", best_params["Ridge"], "\n", cv_rows[-1], "\n", test_rows[-1])


# ======================================================================
# 4. Random Forest - tuned with RandomizedSearchCV
# ======================================================================
print("\n--- Random Forest ---")
rf = Pipeline([("prep", preprocessor(TARGET)),
               ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])

rf_grid = {
    "model__n_estimators": [100, 200, 400],          # number of trees
    "model__max_depth": [10, 20, 30, None],          # how deep each tree may grow
    "model__min_samples_leaf": [1, 2, 5, 10],        # min. cars per leaf (higher = smoother)
    "model__max_features": [0.3, 0.5, 0.8, "sqrt"],  # share of features tried at each split
}
rf_search = RandomizedSearchCV(rf, rf_grid, n_iter=N_ITER_RF, cv=cv, scoring=SCORING, refit="r2",
                               return_train_score=True, random_state=RANDOM_STATE,
                               n_jobs=1)            # the forest itself already uses all cores
rf_search.fit(X_train, y_train)

cv_rows.append(search_summary("Random Forest", rf_search))
best_params["Random Forest"] = clean_params(rf_search.best_params_)
test_rows.append(test_summary("Random Forest", rf_search.best_estimator_))
joblib.dump(rf_search.best_estimator_, "models/rf_best.joblib", compress=3)   # compressed: RF files are large
pd.DataFrame(rf_search.cv_results_).to_csv("results/rf_tuning.csv", index=False)
print("best:", best_params["Random Forest"], "\n", cv_rows[-1], "\n", test_rows[-1])
# I saved the models for reproducability in joblib formart. (instead of pickle).

# ======================================================================
# 5. XGBoost - tuned with RandomizedSearchCV
# ======================================================================
print("\n--- XGBoost ---")
xgb = Pipeline([("prep", preprocessor(TARGET)),
                ("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"))])

xgb_grid = {
    "model__n_estimators": [300, 600, 1000],         # number of boosting rounds (trees)
    "model__max_depth": [4, 6, 8, 10],               # depth of each tree
    "model__learning_rate": [0.03, 0.05, 0.1],       # how much each new tree corrects
    "model__subsample": [0.7, 0.85, 1.0],            # share of rows used per tree
    "model__colsample_bytree": [0.6, 0.8, 1.0],      # share of features used per tree
    "model__min_child_weight": [1, 5, 10],           # min. weight in a leaf (higher = less overfitting)
}
xgb_search = RandomizedSearchCV(xgb, xgb_grid, n_iter=N_ITER_XGB, cv=cv, scoring=SCORING, refit="r2",
                                return_train_score=True, random_state=RANDOM_STATE,
                                n_jobs=1)           # XGBoost itself already uses all cores
xgb_search.fit(X_train, y_train)

cv_rows.append(search_summary("XGBoost", xgb_search))
best_params["XGBoost"] = clean_params(xgb_search.best_params_)
test_rows.append(test_summary("XGBoost", xgb_search.best_estimator_))
joblib.dump(xgb_search.best_estimator_, "models/xgb_best.joblib")
pd.DataFrame(xgb_search.cv_results_).to_csv("results/xgb_tuning.csv", index=False)
print("best:", best_params["XGBoost"], "\n", cv_rows[-1], "\n", test_rows[-1])


# ======================================================================
# 6. Summary: choose the final model on CV, then show all test scores
# ======================================================================
cv_table = pd.DataFrame(cv_rows)
cv_table["gap_train_minus_CV"] = cv_table["train_R2_log"] - cv_table["CV_R2_log"]   # overfitting check
test_table = pd.DataFrame(test_rows)

final_model = cv_table.loc[cv_table["CV_R2_log"].idxmax(), "model"]   # decided on CV, not on test

print("\n=== Cross-validation (training set) ===")
print(cv_table.round(4).to_string(index=False))
print("\n=== Test set (used once per model) ===")
print(test_table.round(4).to_string(index=False))
print(f"\nFinal model (highest CV R2): {final_model}")


# ======================================================================
# 7. Save everything needed for the plots and the report
# ======================================================================
cv_table.to_csv("results/cv_scores.csv", index=False)
test_table.to_csv("results/test_scores.csv", index=False)
predictions.to_csv("results/test_predictions.csv", index=False)
with open("results/best_params.json", "w") as f:
    json.dump({"final_model": final_model, "params": best_params}, f, indent=2)

print("\nSaved: models/*.joblib, results/cv_scores.csv, results/test_scores.csv, "
      "results/test_predictions.csv, results/rf_tuning.csv, results/xgb_tuning.csv, results/best_params.json")