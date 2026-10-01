
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
DATA = ROOT / "car_details_v3.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42


raw = pd.read_csv(DATA)
expected = {
    "name", "year", "selling_price", "km_driven", "fuel", "seller_type",
    "transmission", "owner", "mileage", "engine", "max_power", "torque", "seats"
}
if not expected.issubset(raw.columns):
    raise ValueError(f"Missing columns: {sorted(expected - set(raw.columns))}")

audit = {
    "raw_rows": int(len(raw)),
    "raw_columns": int(raw.shape[1]),
    "exact_duplicates": int(raw.duplicated().sum()),
    "missing_by_column": {k: int(v) for k, v in raw.isna().sum().items()},
    "year_range": [int(raw.year.min()), int(raw.year.max())],
    "price_range_inr": [int(raw.selling_price.min()), int(raw.selling_price.max())],
}
print("RAW DATA AUDIT")
print(json.dumps(audit, indent=2))



data = raw.drop_duplicates().copy()
valid = (
    data["selling_price"].gt(0)
    & data["year"].between(1980, 2020)
    & data["km_driven"].ge(0)
    & data["name"].notna()
)
data = data.loc[valid].copy()
data["brand"] = data["name"].str.split().str[0].str.title()
data["engine_cc"] = pd.to_numeric(
    data["engine"].astype("string").str.extract(r"(\d+(?:\.\d+)?)", expand=False),
    errors="coerce",
)
data["power_bhp"] = pd.to_numeric(
    data["max_power"].astype("string").str.extract(r"(\d+(?:\.\d+)?)", expand=False),
    errors="coerce",
)
data.loc[~data["engine_cc"].between(500, 7000), "engine_cc"] = np.nan
data.loc[~data["power_bhp"].between(10, 1000), "power_bhp"] = np.nan
data["log_km"] = np.log1p(data["km_driven"])

numeric = ["year", "log_km", "engine_cc", "power_bhp", "seats"]
categorical = ["brand", "fuel", "seller_type", "transmission", "owner"]
features = numeric + categorical
X = data[features]
y = data["selling_price"]

X_develop, X_test, y_develop, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_develop, y_develop, test_size=0.15 / 0.85, random_state=RANDOM_STATE
)
assert len(X_train) + len(X_val) + len(X_test) == len(data)
assert not (set(X_train.index) & set(X_val.index))
assert not (set(X_train.index) & set(X_test.index))
assert not (set(X_val.index) & set(X_test.index))


eda = data.loc[X_train.index].copy()
plt.rcParams.update({"figure.dpi": 135, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.hist(eda["selling_price"], bins=np.logspace(np.log10(eda.selling_price.min()), np.log10(eda.selling_price.max()), 32), color="#176B82")
ax.set_xscale("log")
ax.set(title="Training listing prices", xlabel="Advertised price (INR, log scale)", ylabel="Listings")
fig.tight_layout(); fig.savefig(OUT / "eda_price.png"); plt.close(fig)

year_stats = eda.groupby("year")["selling_price"].agg(["median", "size"])
year_stats = year_stats.loc[year_stats["size"] >= 10]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(year_stats.index, year_stats["median"], marker="o", markersize=3, color="#176B82")
ax.set(title="Median asking price by model year", xlabel="Model year", ylabel="Median price (INR)")
fig.tight_layout(); fig.savefig(OUT / "eda_year.png"); plt.close(fig)

sample = eda.sample(min(1500, len(eda)), random_state=RANDOM_STATE)
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.scatter(sample["km_driven"], sample["selling_price"], s=12, alpha=0.3, color="#176B82")
ax.set(xlim=(0, 300000), ylim=(0, 3000000), title="Mileage and asking price (sample of training listings)", xlabel="Kilometres driven", ylabel="Advertised price (INR)")
fig.tight_layout(); fig.savefig(OUT / "eda_km.png"); plt.close(fig)

major_fuels = [f for f in ["Petrol", "Diesel", "CNG", "LPG"] if (eda.fuel == f).sum() >= 20]
plot_price = eda.loc[eda.selling_price <= eda.selling_price.quantile(0.99)]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.boxplot([plot_price.loc[plot_price.fuel == f, "selling_price"] for f in major_fuels], tick_labels=major_fuels, showfliers=False)
ax.set(title="Asking price by fuel type", xlabel="Fuel", ylabel="Advertised price (INR, top 1% hidden in plot)")
fig.tight_layout(); fig.savefig(OUT / "eda_fuel.png"); plt.close(fig)

eda_notes = {
    "price_median_inr": float(eda.selling_price.median()),
    "price_mean_inr": float(eda.selling_price.mean()),
    "year_price_spearman": float(eda[["year", "selling_price"]].corr(method="spearman").iloc[0, 1]),
    "km_price_spearman": float(eda[["km_driven", "selling_price"]].corr(method="spearman").iloc[0, 1]),
    "fuel_median_inr": {str(k): float(v) for k, v in eda.groupby("fuel").selling_price.median().items()},
}
print("\nEDA INTERPRETATION")
print("Price mean exceeds median, showing a long high-price tail.")
print(f"Newer model years generally have higher prices (Spearman rho={eda_notes['year_price_spearman']:.2f}).")
print(f"Higher mileage generally accompanies lower prices (Spearman rho={eda_notes['km_price_spearman']:.2f}).")
print("Fuel groups have different price distributions, but this is descriptive, not a causal fuel effect.")



preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ]
)

def scores(y_true, prediction):
    return {
        "mae_inr": float(mean_absolute_error(y_true, prediction)),
        "rmse_inr": float(np.sqrt(mean_squared_error(y_true, prediction))),
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
print("\nTREE CV", tree_search.best_params_, "CV MAE", -tree_search.best_score_)

validation = {}
for name, model in models.items():
    if name != "Decision tree":
        model.fit(X_train, y_train)
    validation[name] = scores(y_val, model.predict(X_val))
print("\nVALIDATION RESULTS")
print(pd.DataFrame(validation).T.round(3).to_string())

winner_name = min(validation, key=lambda name: validation[name]["mae_inr"])
winner = models[winner_name]
test_predictions = winner.predict(X_test)
test_results = scores(y_test, test_predictions)
baseline_test = scores(y_test, models["Median baseline"].predict(X_test))
success = bool(
    test_results["mae_inr"] < 150000
    and test_results["mae_inr"] <= 0.8 * baseline_test["mae_inr"]
)
print("\nSELECTED MODEL", winner_name)
print("HELD-OUT TEST", json.dumps(test_results, indent=2))
print("BASELINE TEST MAE", round(baseline_test["mae_inr"]))
print("MEETS SUCCESS CRITERION", success)


cuts = np.quantile(y_train, [0.25, 0.5, 0.75])
band_labels = ["lowest quarter", "lower middle", "upper middle", "highest quarter"]
test_bands = pd.cut(y_test, bins=[-np.inf, *cuts, np.inf], labels=band_labels, include_lowest=True)
absolute_errors = np.abs(y_test.to_numpy() - test_predictions)
error_by_band = (
    pd.DataFrame({"band": test_bands.to_numpy(), "absolute_error": absolute_errors})
    .groupby("band", observed=True)["absolute_error"]
    .agg(["size", "median", "mean"])
)
print("\nTEST ERROR BY PRICE BAND")
print(error_by_band.round(0).to_string())

fig, ax = plt.subplots(figsize=(6.5, 5.6))
ax.scatter(y_test, test_predictions, s=13, alpha=0.35, color="#176B82")
limit = float(max(y_test.max(), test_predictions.max()))
ax.plot([0, limit], [0, limit], color="#DB6B4A", linewidth=1.5)
ax.set(xlabel="Actual advertised price (INR)", ylabel="Predicted price (INR)", title=f"Test predictions: {winner_name}")
fig.tight_layout(); fig.savefig(OUT / "test_predictions.png"); plt.close(fig)

report = {
    "source": "https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho",
    "mirror": "https://github.com/HarshKapadia2/car-details/blob/main/data/car_details_v3.csv",
    "audit": audit,
    "clean_rows": int(len(data)),
    "split_rows": {"train": int(len(X_train)), "validation": int(len(X_val)), "test": int(len(X_test))},
    "feature_columns": features,
    "eda": eda_notes,
    "year_medians": {
        str(int(k)): float(v)
        for k, v in year_stats.loc[year_stats.index >= 2000, "median"].items()
    },
    "tree_cv": {"best_parameters": tree_search.best_params_, "mean_mae_inr": float(-tree_search.best_score_), "folds": 3},
    "validation": validation,
    "selected_model": winner_name,
    "test": test_results,
    "baseline_test": baseline_test,
    "success_criterion_met": success,
    "error_by_price_band": error_by_band.reset_index().to_dict(orient="records"),
}
(OUT / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print("\nSaved charts and metrics in", OUT)

