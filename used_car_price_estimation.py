# %% [markdown]
# # Used car price estimation in Kazakhstan
# **Team:** Alisher Akhmet and Aslan Muratov
#
# Regression target: advertised price of a used car in Kazakhstani tenge (KZT).
# Main metric: mean absolute error (MAE) in KZT. Success criterion: the selected
# model must reduce held-out test MAE by at least 20% against a training-median
# baseline. RMSE and R2 are secondary metrics.
#
# Dataset: Darkhan Mutashev, Kolesa-Cars-2025, Kaggle (MIT licence),
# https://www.kaggle.com/datasets/mutashevdarkhan/kolesa-cars-2025
# Public dataset last updated 2025-12-18. The source rows link to Kolesa.kz
# listings; the collection dates of individual rows are not provided.
# Prices are asking prices, not verified sale prices. This is a 2025 snapshot,
# so the result is not a live valuation of the 2026 market.
#
# Run from this folder: `python used_car_price_estimation.py`

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "kolesa_cars_2025.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
SOURCE = "https://www.kaggle.com/datasets/mutashevdarkhan/kolesa-cars-2025"


# %% [markdown]
# ## 1. Data acquisition and audit
# The CSV is the unmodified `cars_kolesa_parsed.csv` from the Kaggle download,
# renamed for clarity. It has prices in KZT, listing URLs, model year, mileage,
# engine size and categories. We verify the currency symbol and report the
# source's missing fields before modelling.

# %%
raw = pd.read_csv(DATA)
expected = {
    "name", "price", "price_raw", "desc", "link", "brand", "model",
    "condition", "year", "mileage_km", "engine_liters", "fuel",
    "transmission", "drive"
}
if not expected.issubset(raw.columns):
    raise ValueError(f"Missing columns: {sorted(expected - set(raw.columns))}")
if not raw["price_raw"].astype("string").str.contains("₸", regex=False).all():
    raise ValueError("Some source price labels are not shown in tenge")
raw["ad_id"] = raw["link"].astype("string").str.extract(r"/show/([0-9]+)", expand=False)
if raw["ad_id"].isna().any():
    raise ValueError("Some Kolesa listing IDs could not be read")

audit = {
    "raw_rows": int(len(raw)),
    "raw_columns": 14,
    "raw_unique_ads": int(raw.ad_id.nunique()),
    "repeated_ad_rows": int(raw.ad_id.duplicated().sum()),
    "new_car_rows_excluded": int(raw.condition.eq("new").sum()),
    "raw_missing_by_column": {k: int(v) for k, v in raw.drop(columns="ad_id").isna().sum().items()},
    "used_year_range": [int(raw.loc[raw.condition.eq("used"), "year"].min()),
                        int(raw.loc[raw.condition.eq("used"), "year"].max())],
    "all_prices_labelled_kzt": True,
}
print("RAW DATA AUDIT")
print(json.dumps(audit, indent=2, ensure_ascii=False))


# %% [markdown]
# ## 2. Cleaning, features and split
# Keep used listings only. An ad can appear more than once due to source search
# pages; deduplicate by the actual Kolesa ad ID before splitting. Remove
# listings explicitly marked "on order" because they are not local available
# cars. Reject nonpositive prices, invalid years and negative mileage. Preserve
# high-price observations so error analysis reflects the real price range.
#
# The source parser often missed engine size even when it occurs in the spec
# text, so we recover it from `desc` when available. We also extract body type,
# keep model text, and use log1p(km) to reduce mileage skew. No target-derived
# feature, listing ID or raw price label is passed to any model.

# %%
used = raw.loc[raw.condition.eq("used")].copy()
on_order = used["desc"].astype("string").str.contains("На заказ", case=False, na=False)
used = used.loc[~on_order].drop_duplicates(subset="ad_id").copy()
valid = (
    used["price"].gt(0)
    & used["year"].between(1950, 2025)
    & (used["mileage_km"].isna() | used["mileage_km"].ge(0))
    & used["brand"].notna()
)
used = used.loc[valid].copy()

engine_from_text = pd.to_numeric(
    used["desc"].astype("string")
    .str.extract(r"([0-9]+(?:[.,][0-9]+)?)\s*л", expand=False)
    .str.replace(",", ".", regex=False),
    errors="coerce",
)
used["engine_liters_clean"] = used["engine_liters"].combine_first(engine_from_text)
used.loc[~used["engine_liters_clean"].between(0.5, 8.0), "engine_liters_clean"] = np.nan
used["body"] = (
    used["desc"].astype("string")
    .str.extract(r"Б/у\s+([^,]+)", expand=False)
    .str.strip().str.lower()
)
used["brand"] = used["brand"].astype("string").str.strip()
used["model"] = used["model"].astype("string").str.strip()
used["log_km"] = np.log1p(used["mileage_km"])
used["mileage_missing"] = used["mileage_km"].isna().astype(int)

numeric = ["year", "log_km", "mileage_missing", "engine_liters_clean"]
categorical = ["brand", "model", "body", "fuel", "transmission"]
features = numeric + categorical
X = used[features]
y = used["price"]

X_develop, X_test, y_develop, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_develop, y_develop, test_size=0.15 / 0.85, random_state=RANDOM_STATE
)
assert len(X_train) + len(X_val) + len(X_test) == len(used)
assert not (set(X_train.index) & set(X_val.index))
assert not (set(X_train.index) & set(X_test.index))
assert not (set(X_val.index) & set(X_test.index))


# %% [markdown]
# ## 3. Four training-only EDA plots
# Plot 1: price distribution. Plot 2: median price by model year. Plot 3:
# mileage versus price. Plot 4: price by fuel type. All use training rows only
# so the final test labels do not inform modelling decisions. The fuel chart
# is descriptive; different car mixes can explain group differences.

# %%
eda = used.loc[X_train.index].copy()
plt.rcParams.update({"figure.dpi": 135, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.hist(eda.price, bins=np.logspace(np.log10(eda.price.min()), np.log10(eda.price.max()), 30), color="#176B82")
ax.set_xscale("log")
ax.set(title="Used-car asking prices in Kazakhstan (training)", xlabel="Asking price (KZT, log scale)", ylabel="Listings")
fig.tight_layout(); fig.savefig(OUT / "eda_price_kzt.png"); plt.close(fig)

year_stats = eda.groupby("year").price.agg(["median", "size"])
year_stats = year_stats.loc[(year_stats.index >= 1990) & (year_stats["size"] >= 8)]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(year_stats.index, year_stats["median"] / 1e6, marker="o", markersize=3, color="#176B82")
ax.set(title="Median asking price by model year", xlabel="Model year", ylabel="Median asking price (million KZT)")
fig.tight_layout(); fig.savefig(OUT / "eda_year_kzt.png"); plt.close(fig)

sample = eda.loc[eda.mileage_km.notna()].sample(min(1200, eda.mileage_km.notna().sum()), random_state=RANDOM_STATE)
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.scatter(sample.mileage_km, sample.price / 1e6, s=12, alpha=0.3, color="#176B82")
ax.set(xlim=(0, 500000), title="Mileage and asking price (training sample)", xlabel="Kilometres driven", ylabel="Asking price (million KZT)")
fig.tight_layout(); fig.savefig(OUT / "eda_km_kzt.png"); plt.close(fig)

fuels = [f for f in ["petrol", "hybrid", "diesel", "gas"] if (eda.fuel == f).sum() >= 15]
plot_price = eda.loc[eda.price <= eda.price.quantile(0.99)]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.boxplot([plot_price.loc[plot_price.fuel == f, "price"] / 1e6 for f in fuels], tick_labels=fuels, showfliers=False)
ax.set(title="Asking price by fuel type", xlabel="Fuel", ylabel="Asking price (million KZT; top 1% hidden)")
fig.tight_layout(); fig.savefig(OUT / "eda_fuel_kzt.png"); plt.close(fig)

eda_notes = {
    "training_price_median_kzt": float(eda.price.median()),
    "training_price_mean_kzt": float(eda.price.mean()),
    "year_price_spearman": float(eda[["year", "price"]].corr(method="spearman").iloc[0, 1]),
    "km_price_spearman": float(eda[["mileage_km", "price"]].corr(method="spearman").iloc[0, 1]),
    "fuel_medians_kzt": {str(k): float(v) for k, v in eda.groupby("fuel").price.median().items()},
    "fuel_counts": {str(k): int(v) for k, v in eda.fuel.value_counts().items()},
}
print("\nEDA INTERPRETATION")
print("The price mean exceeds the median because expensive listings form a long tail.")
print(f"Newer cars generally list for more (Spearman rho={eda_notes['year_price_spearman']:.2f}).")
print(f"Higher mileage generally accompanies lower asking prices (rho={eda_notes['km_price_spearman']:.2f}).")
print("Fuel groups differ, but those differences are not causal estimates.")


# %% [markdown]
# ## 4. Baseline and three lecture algorithms
# The numeric median imputer and scaler, plus categorical mode imputer and
# one-hot encoder, are fitted within each training fold. Rare categories are
# grouped by OneHotEncoder so a singleton trim cannot dominate the model.
# We compare the training-median baseline, linear regression, KNN and a
# decision tree. The tree is tuned by 3-fold CV on training rows. Validation
# MAE selects the winner. The test set is evaluated only after selection.

# %%
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5))]), categorical),
    ]
)

def scores(y_true, prediction):
    return {
        "mae_kzt": float(mean_absolute_error(y_true, prediction)),
        "rmse_kzt": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
    }

models = {
    "Median baseline": DummyRegressor(strategy="median"),
    "Linear regression": Pipeline([("prep", preprocess), ("model", LinearRegression())]),
    "KNN regression": Pipeline([("prep", preprocess), ("model", KNeighborsRegressor(n_neighbors=15, weights="distance"))]),
}
tree_search = GridSearchCV(
    Pipeline([("prep", preprocess), ("model", DecisionTreeRegressor(random_state=RANDOM_STATE))]),
    param_grid={"model__max_depth": [6, 10, 14], "model__min_samples_leaf": [3, 10]},
    scoring="neg_mean_absolute_error", cv=3, n_jobs=-1, refit=True,
)
tree_search.fit(X_train, y_train)
models["Decision tree"] = tree_search.best_estimator_
print("\nTREE CV", tree_search.best_params_, "CV MAE KZT", -tree_search.best_score_)

validation = {}
for name, model in models.items():
    if name != "Decision tree":
        model.fit(X_train, y_train)
    validation[name] = scores(y_val, model.predict(X_val))
print("\nVALIDATION RESULTS (KZT)")
print(pd.DataFrame(validation).T.round(2).to_string())

winner_name = min(validation, key=lambda name: validation[name]["mae_kzt"])
winner = models[winner_name]
test_predictions = winner.predict(X_test)
test_results = scores(y_test, test_predictions)
baseline_test = scores(y_test, models["Median baseline"].predict(X_test))
improvement = 1 - test_results["mae_kzt"] / baseline_test["mae_kzt"]
success = bool(improvement >= 0.20)
print("\nSELECTED MODEL", winner_name)
print("HELD-OUT TEST", json.dumps(test_results, indent=2))
print("BASELINE TEST MAE KZT", round(baseline_test["mae_kzt"]))
print("TEST MAE IMPROVEMENT", f"{improvement:.1%}")
print("MEETS SUCCESS CRITERION", success)


# %% [markdown]
# ## 5. Initial error analysis and limits
# Price-band boundaries come from training labels only. We evaluate test
# absolute errors within those bands. The dataset has no city, listing date,
# verified transaction price or detailed condition measure. The published
# dataset was last updated in December 2025 and should not be described as
# live 2026 market data.

# %%
cuts = np.quantile(y_train, [0.25, 0.5, 0.75])
band_labels = ["lowest quarter", "lower middle", "upper middle", "highest quarter"]
test_bands = pd.cut(y_test, bins=[-np.inf, *cuts, np.inf], labels=band_labels, include_lowest=True)
absolute_errors = np.abs(y_test.to_numpy() - test_predictions)
error_by_band = (
    pd.DataFrame({"band": test_bands.to_numpy(), "absolute_error": absolute_errors})
    .groupby("band", observed=True)["absolute_error"].agg(["size", "median", "mean"])
)
print("\nTEST ERROR BY PRICE BAND (KZT)")
print(error_by_band.round(0).to_string())

fig, ax = plt.subplots(figsize=(6.5, 5.6))
ax.scatter(y_test / 1e6, test_predictions / 1e6, s=13, alpha=0.35, color="#176B82")
limit = float(max(y_test.max(), test_predictions.max()) / 1e6)
ax.plot([0, limit], [0, limit], color="#DB6B4A", linewidth=1.5)
ax.set(xlabel="Actual asking price (million KZT)", ylabel="Predicted price (million KZT)", title=f"Test predictions: {winner_name}")
fig.tight_layout(); fig.savefig(OUT / "test_predictions_kzt.png"); plt.close(fig)

report = {
    "dataset_url": SOURCE,
    "dataset_last_updated": "2025-12-18",
    "dataset_license": "MIT (Kaggle catalog metadata)",
    "currency": "KZT",
    "target": "advertised asking price of used cars",
    "audit": audit,
    "on_order_used_rows_excluded": int(on_order.sum()),
    "clean_unique_used_rows": int(len(used)),
    "remaining_missing_engine_liters": int(used.engine_liters_clean.isna().sum()),
    "remaining_missing_mileage_km": int(used.mileage_km.isna().sum()),
    "split_rows": {"train": int(len(X_train)), "validation": int(len(X_val)), "test": int(len(X_test))},
    "feature_columns": features,
    "eda": eda_notes,
    "year_medians_kzt": {str(int(k)): float(v) for k, v in year_stats["median"].items()},
    "tree_cv": {"best_parameters": tree_search.best_params_, "mean_mae_kzt": float(-tree_search.best_score_), "folds": 3},
    "validation": validation,
    "selected_model": winner_name,
    "test": test_results,
    "baseline_test": baseline_test,
    "test_mae_improvement_fraction": float(improvement),
    "success_criterion_met": success,
    "error_by_price_band": error_by_band.reset_index().to_dict(orient="records"),
}
(OUT / "metrics.json").write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
print("\nSaved Kazakhstan charts and metrics in", OUT)

# %% [markdown]
# ## 6. Final-stage plan
# 1. Obtain a dated 2026 snapshot with permission and evaluate time drift.
# 2. Add city and car condition if a licensed source provides them.
# 3. Compare stronger ensembles using training-only cross-validation.
# 4. Report segment errors and uncertainty, especially for expensive cars.
