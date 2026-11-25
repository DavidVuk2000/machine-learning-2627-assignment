# Regression plots + SHAP - made only from the saved files of regression_final.py (no retraining).
# Run it after regression_final.py (duplicates-removed run). All figures go to Figures/regression/.
# Needs: python -m pip install shap

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score
import shap

from Data_cleaning import split_data                     # same cleaning the models were trained with
import Data_cleaning_with_dups as cleaning_dups          # cleaning version that keeps the duplicates

RESULT_DIR = "results_dedup"            # results of the run without duplicates (the final one)
MODEL_DIR = "models_dedup"              # models of that same run
DUPS_RESULT_DIR = "results_with_dups"   # saved run that kept the duplicates (only for the boxplot)
DUPS_MODEL_DIR = "models_with_dups"
OUT = "Figures/regression"              # where the figures are saved
N_SAMPLE_XGB = 3000                     # test cars explained by SHAP for XGBoost
N_SAMPLE_RF = 300                       # fewer for RF: its trees are fully grown, so SHAP is much slower
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9c9a92"
MODELS = ["Dummy", "Ridge", "Random Forest", "XGBoost"]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)       # remove top/right borders
    ax.grid(color="#e5e4df", linewidth=0.6)               # light grid
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=150)
    plt.close(fig)
    print("saved", f"{OUT}/{name}.png")


# Load the saved results

cv = pd.read_csv(f"{RESULT_DIR}/cv_scores.csv").set_index("model").loc[MODELS]
test = pd.read_csv(f"{RESULT_DIR}/test_scores.csv").set_index("model").loc[MODELS]
preds = pd.read_csv(f"{RESULT_DIR}/test_predictions.csv")
with open(f"{RESULT_DIR}/best_params.json") as f:
    FINAL = json.load(f)["final_model"]               # model chosen on CV, e.g. "XGBoost"
print("Final model:", FINAL)

true_price = preds["y_true_$"]
pred_price = preds[f"{FINAL}_pred_$"]


# 1. Predicted vs actual price (final model) - required by the assignment

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(true_price, pred_price, s=3, alpha=0.25, color=BLUE, linewidths=0)
lims = [min(true_price.min(), pred_price.min()), max(true_price.max(), pred_price.max())]
ax.plot(lims, lims, color=ORANGE, linewidth=1.5, label="perfect prediction")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(frameon=False)
style(ax, f"{FINAL} - predicted vs actual (test set)", "Actual price ($)", "Predicted price ($)")
save(fig, "01_predicted_vs_actual")


# 3. Error per price range (final model): MAE in each price bin

bins = [500, 5_000, 10_000, 20_000, 40_000, 80_000, 300_000]
labels = ["0.5-5k", "5-10k", "10-20k", "20-40k", "40-80k", "80-300k"]
price_bin = pd.cut(true_price, bins=bins, labels=labels, include_lowest=True)
abs_err = (pred_price - true_price).abs()
per_bin = pd.DataFrame({"bin": price_bin, "err": abs_err}).groupby("bin", observed=False)["err"].agg(["mean", "size"])
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(per_bin.index.astype(str), per_bin["mean"], color=BLUE)
for bar, n in zip(bars, per_bin["size"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}",
            ha="center", va="bottom", fontsize=9, color="#555")
style(ax, f"{FINAL} - mean absolute error per price range (test set)", "Actual price range ($)", "MAE ($)")
save(fig, "03_error_per_price_range")


# 4. CV R2 vs test R2 for all models

x = np.arange(len(MODELS))
fig, ax = plt.subplots(figsize=(8, 5))
cv_bars = ax.bar(x - 0.2, cv["CV_R2_log"], 0.4, yerr=cv["CV_R2_std"], color=BLUE, label="cross-validation")
test_bars = ax.bar(x + 0.2, test["test_R2_log"], 0.4, color=ORANGE, label="test set")
ax.bar_label(cv_bars, fmt="%.3f", padding=3, fontsize=9)       # value above each bar
ax.bar_label(test_bars, fmt="%.3f", padding=3, fontsize=9)
ax.set_ylim(top=1.0)                                            # room for the labels
ax.set_xticks(x, MODELS)
ax.axhline(0, color=GREY, linewidth=0.8)
ax.legend(frameon=False)
style(ax, "R² on log(price) - cross-validation vs test", "", "R²")
save(fig, "04_cv_vs_test_r2")


# 4b. Boxplot of the 5 CV fold scores: duplicates removed vs kept
#     (Dummy left out: it sits at R2 ~ 0)

CV = KFold(n_splits=5, shuffle=True, random_state=42)        # same folds as in regression_final.py
BOX_MODELS = ["Ridge", "Random Forest", "XGBoost"]
VERSIONS = [("Duplicates removed", RESULT_DIR, MODEL_DIR, split_data, BLUE),
            ("Duplicates kept", DUPS_RESULT_DIR, DUPS_MODEL_DIR, cleaning_dups.split_data, ORANGE)]


def fold_scores(result_dir, model_dir, split_fn):
    """R2 of each of the 5 CV folds for the best setting of every model."""
    X_tr, _, y_tr, _ = split_fn(pd.read_csv("data.csv"), "sale_price")
    scores = {}
    # Ridge: per-fold scores were not saved, so recompute them (fast) with the saved best settings
    ridge = clone(joblib.load(f"{model_dir}/ridge.joblib"))   # same settings, untrained
    scores["Ridge"] = cross_val_score(ridge, X_tr, y_tr, cv=CV, scoring="r2", n_jobs=-1)
    # RF and XGBoost: per-fold scores of the best combination are in the tuning files
    for name, file in [("Random Forest", "rf_tuning"), ("XGBoost", "xgb_tuning")]:
        tune = pd.read_csv(f"{result_dir}/{file}.csv")
        best = tune.loc[tune["rank_test_r2"] == 1].iloc[0]
        scores[name] = best[[f"split{i}_test_r2" for i in range(5)]].astype(float).values
    return scores


fig, ax = plt.subplots(figsize=(9, 5.5))
for v, (label, res_dir, mod_dir, split_fn, color) in enumerate(VERSIONS):
    if not os.path.exists(res_dir):
        print("skipped", label, "- folder", res_dir, "not found")
        continue
    scores = fold_scores(res_dir, mod_dir, split_fn)
    positions = np.arange(len(BOX_MODELS)) * 3 + v                       # pairs of boxes per model
    data = [scores[m] for m in BOX_MODELS]
    ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
               boxprops={"facecolor": color, "alpha": 0.6, "color": color},
               medianprops={"color": "black"}, whiskerprops={"color": color}, capprops={"color": color},
               flierprops={"markeredgecolor": color})
    for pos, s in zip(positions, data):                                   # mean +- std above each box
        ax.text(pos, s.max() + 0.006, f"{s.mean():.3f}\n± {s.std():.3f}", ha="center", va="bottom", fontsize=8)
    ax.plot([], [], "s", color=color, alpha=0.6, label=label)             # legend entry
ax.set_xticks(np.arange(len(BOX_MODELS)) * 3 + 0.5, BOX_MODELS)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)   # below the plot
ax.margins(y=0.15)
style(ax, "Cross-validation R² per fold (5 folds)", "", "R² on log(price)")
save(fig, "04b_cv_boxplot_duplicates")


# 5. Train vs CV R2 (overfitting gap)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 0.2, cv["train_R2_log"], 0.4, color=GREY, label="train")
ax.bar(x + 0.2, cv["CV_R2_log"], 0.4, color=BLUE, label="cross-validation")
for i, gap in enumerate(cv["train_R2_log"] - cv["CV_R2_log"]):
    ax.text(i, max(cv["train_R2_log"].iloc[i], cv["CV_R2_log"].iloc[i]) + 0.01, f"gap {gap:.2f}",
            ha="center", fontsize=9, color="#555")
ax.set_xticks(x, MODELS)
ax.axhline(0, color=GREY, linewidth=0.8)
ax.set_ylim(top=1.08)                                       # room for the legend and gap labels
ax.legend(frameon=False, loc="upper left", ncol=2)
style(ax, "Train vs cross-validation R² (overfitting check)", "", "R² on log(price)")
save(fig, "05_train_vs_cv_r2")


# 6. Test MAE and RMSE in dollars for all models

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 0.2, test["test_MAE_$"], 0.4, color=BLUE, label="MAE")
ax.bar(x + 0.2, test["test_RMSE_$"], 0.4, color=ORANGE, label="RMSE")
ax.set_xticks(x, MODELS)
ax.legend(frameon=False)
style(ax, "Test error in dollars", "", "$")
save(fig, "06_test_mae_rmse")


# 7-8. Tuning results: CV R2 of every tried combination

for i, (name, file) in enumerate([("Random Forest", "rf_tuning"), ("XGBoost", "xgb_tuning")], start=7):
    tune = pd.read_csv(f"{RESULT_DIR}/{file}.csv").sort_values("mean_test_r2").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(tune.index, tune["mean_test_r2"], yerr=tune["std_test_r2"], fmt="o",
                color=BLUE, ecolor="#86b6ef", capsize=3, label="cross-validation (± std)")
    ax.plot(tune.index, tune["mean_train_r2"], "o", color=GREY, label="train")
    ax.legend(frameon=False)
    ax.set_xticks(tune.index)                                   # one tick per combination
    style(ax, f"{name} - all tried hyperparameter combinations", "Combination (sorted by CV R²)", "R² on log(price)")
    save(fig, f"0{i}_{file}")


# 9. Permutation importance of the final model (original features, test set)

raw = pd.read_csv("data.csv")
X_train, X_test, y_train, y_test = split_data(raw, "sale_price")    # same split as training, no training
model_file = {"XGBoost": "xgb_best", "Random Forest": "rf_best", "Ridge": "ridge", "Dummy": "dummy"}[FINAL]
model = joblib.load(f"{MODEL_DIR}/{model_file}.joblib")

perm = permutation_importance(model, X_test, y_test, scoring="r2", n_repeats=5,
                              random_state=42, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values()
imp_std = pd.Series(perm.importances_std, index=X_test.columns).loc[imp.index]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp.index, imp.values, xerr=imp_std.values, color=BLUE)
style(ax, f"{FINAL} - permutation importance (test set)", "Drop in R² when the feature is shuffled", "")
ax.grid(axis="y", visible=False)
save(fig, "09_permutation_importance")
imp.sort_values(ascending=False).to_csv(f"{RESULT_DIR}/permutation_importance.csv")


# 10. Built-in feature importance of the final tree model (after encoding)

if FINAL in ["XGBoost", "Random Forest"]:
    names = model.named_steps["prep"].get_feature_names_out()
    fi = pd.Series(model.named_steps["model"].feature_importances_, index=names).sort_values().tail(20)
    fi.index = [n.split("__", 1)[-1] for n in fi.index]                 # drop the "num__"/"low__" prefixes
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(fi.index, fi.values, color=BLUE)
    style(ax, f"{FINAL} - built-in feature importance (top 20)", "Importance", "")
    ax.grid(axis="y", visible=False)
    save(fig, "10_builtin_feature_importance")


# 11-12. SHAP beeswarm: which features push the predicted price up (right) or down (left)

def shap_beeswarm(pipeline, n_sample, title, filename, csv_name=None):
    """Explain a sample of test cars with SHAP and save the beeswarm plot."""
    prep, inner_model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    X_sample = X_test.sample(n=min(n_sample, len(X_test)), random_state=42)
    X_enc = pd.DataFrame(prep.transform(X_sample),                   # the numbers the model actually sees
                         columns=[n.split("__", 1)[-1] for n in prep.get_feature_names_out()])
    values = shap.TreeExplainer(inner_model)(X_enc)                  # effect of each feature on each prediction
    shap.plots.beeswarm(values, max_display=20, show=False)
    plt.title(title, loc="left")
    plt.xlabel("SHAP value (effect on predicted log price)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/{filename}.png", dpi=150)
    plt.close()
    print("saved", f"{OUT}/{filename}.png")
    if csv_name:                                                     # mean |SHAP| per feature, for the report
        mean_abs = pd.Series(np.abs(values.values).mean(axis=0), index=X_enc.columns).sort_values(ascending=False)
        mean_abs.to_csv(f"{RESULT_DIR}/{csv_name}.csv")


# 11. XGBoost, the final model
shap_beeswarm(joblib.load(f"{MODEL_DIR}/xgb_best.joblib"), N_SAMPLE_XGB,
              "XGBoost - SHAP values (test set sample)", "11_shap_beeswarm", "shap_mean_abs")

# 12. Random Forest, to check the patterns are not specific to one model
shap_beeswarm(joblib.load(f"{MODEL_DIR}/rf_best.joblib"), N_SAMPLE_RF,
              "Random Forest - SHAP values (test set sample)", "12_shap_beeswarm_rf")

print(f"\nDone - all figures are in {OUT}/")