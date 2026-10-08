# Used car price estimation in Kazakhstan

**Midterm course project**  
**Team:** Alisher Akhmet (Алишер Ахмет) and Aslan Muratov (Муратов Аслан)  
**Topic:** confirmed by the instructor as available, according to the team.

## Project question

Can we estimate the **advertised asking price** of a used car listed in Kazakhstan from its model year, brand, model, mileage, engine size, body type, fuel and transmission? This is a regression task. The target is `price`, measured in **Kazakhstani tenge (KZT, ₸)**. Our primary metric is mean absolute error (MAE), also in tenge. We set a midterm success criterion of at least a 20% reduction in held-out test MAE against the training-median baseline. RMSE and R² provide additional context.

## Dataset and provenance

- Original public dataset: [Darkhan Mutashev, *Kolesa-Cars-2025*, Kaggle](https://www.kaggle.com/datasets/mutashevdarkhan/kolesa-cars-2025). Kaggle lists its licence as **MIT** and its last update as **18 December 2025**.
- Local file: `kolesa_cars_2025.csv`, a renamed, otherwise unmodified copy of the downloaded `cars_kolesa_parsed.csv`. It has **3,999 rows and 14 source columns**. The rows link to individual [Kolesa.kz](https://kolesa.kz/cars/) advertisements in Kazakhstan. Every `price_raw` label uses the ₸ symbol.
- The target is a **listing asking price**, not a verified transaction price. The dataset does not include listing dates or city in its columns. The Kaggle update date is a dataset version date, not an observation date for every ad. This is a Kazakhstan-specific 2025 dataset, **not a live September 2026 price feed**.

## Submission files

- `used_car_price_estimation.py`: runnable Python notebook-style script (`# %%` cells), accepted by the assignment as a `.py` project notebook.
- `kolesa_cars_2025.csv`: source data for offline reproducibility.
- `results/`: four EDA plots, test prediction plot and `metrics.json` from the verified run.
- `requirements.txt`: Python package requirements.

## Setup and run

Use Python 3.10 or newer. From the folder containing these files:

```powershell
python -m pip install -r requirements.txt
python used_car_price_estimation.py
```

The script reads the local CSV and writes `results/` without downloading anything. It uses random seed 42.

## Data understanding and preparation

The original file has 3,999 rows, including **527 new-car rows**. The listing IDs reveal **968 repeated rows** from the source search results. We keep used cars, exclude **51 rows marked “На заказ”** (cars offered on order rather than locally available stock), and deduplicate by listing ID **before** splitting. Basic validity checks retain positive prices, model years 1950–2025 and nonnegative reported mileage. The final modelling set has **2,553 unique used-car listings**.

Source parsing left many engine values blank although an engine size appeared in the description. We recovered engine displacement in litres from that text and left only **17** cleaned listings without it. **497** cleaned listings lack mileage; the model imputes it and receives a missingness indicator. We extract body type from the specification text, use `log1p(mileage_km)` for a skewed numeric feature, and encode categorical car attributes. `drive` is omitted because it is missing in 3,964 of 3,999 source rows. Listing ID, URL, `price_raw` and free-form description are never model inputs.

All imputation, scaling and one-hot encoding occur **inside scikit-learn pipelines**, fitted on training rows or CV folds only. We split into **1,787 training**, **383 validation** and **383 untouched test** rows (approximately 70/15/15). The four EDA plots use only training rows. Three-fold cross-validation on training rows tunes the tree; validation MAE selects the model; the test set is evaluated once after selection.

## EDA findings

The training median asking price is **₸6.70 million** and the mean is **₸9.53 million**, showing a high-price tail. Model year has a positive Spearman association with price (**ρ = 0.69**); kilometres driven has a negative association (**ρ = -0.53**). Fuel-group price distributions differ, but the hybrid and gas groups in training contain only **22** and **24** cars respectively, so their medians are unstable. The four saved plots show price distribution, model year, mileage and fuel type, with interpretations printed by the script.

## Current model results

Validation MAE: training-median baseline **₸6.50m**; linear regression **₸4.65m**; decision tree regression **₸3.44m**; KNN regression **₸3.18m**. Three-fold CV selected tree depth 10 and minimum leaf size 3, with mean CV MAE **₸2.86m**. KNN had the lowest validation MAE and was chosen before inspecting the test set.

On the held-out test set, KNN achieved **MAE ₸2,444,769**, **RMSE ₸5,891,766** and **R² 0.715**. The median baseline's test MAE was **₸5,950,858**. KNN reduced MAE by **58.9%**, meeting the stated midterm criterion. In the highest training-price quartile, mean absolute test error was **₸6.44m**, compared with **₸1.13m** in the lowest quartile. This is a material weakness for expensive cars.

## Team contributions

The team confirmed this division: **Alisher Akhmet** handled dataset sourcing, auditing, cleaning and EDA; **Aslan Muratov** handled model training, cross-validation, metrics and error analysis.

## Limitations and final-stage plan

This dataset is a 2025 public snapshot and may not represent prices in September 2026. It lacks individual listing dates, city, detailed condition and realized sale price. Repeated source ads required deduplication, and the expensive-car segment has high error. The project should next obtain a licensed, dated 2026 snapshot; add city and condition when possible; compare stronger models with training-only CV; and evaluate drift and segment errors on later listings.
