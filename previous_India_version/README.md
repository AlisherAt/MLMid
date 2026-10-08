# Used car price estimation

**Midterm course project**  
**Team:** Alisher Akhmet (Алишер Ахмет) and Aslan Muratov (Муратов Аслан)
**Topic status:** confirmed by the instructor as available, according to the team.

## Project question

Can a model estimate the **advertised** price of a used car in India from its model year, distance driven, brand, fuel, transmission, engine displacement, power and other listing attributes? This is a regression problem. The target, `selling_price`, is measured in Indian rupees (INR). Our primary metric is mean absolute error (MAE), which has the same unit as the price. The midterm success criterion is a held-out test MAE below INR 150,000 and at least 20% below the training-median baseline's test MAE.

## Dataset and acquisition

- Original source: [Nehal Birla, Vehicle dataset (CarDekho) on Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho).
- Local file: `car_details_v3.csv`, downloaded from [this public GitHub mirror](https://github.com/HarshKapadia2/car-details/blob/main/data/car_details_v3.csv). This file contains 8,128 rows and 13 columns. It is a historical collection of used-car listings, not synthetic data.
- Kaggle lists the licence as **Open Database License** for the database and **Database Contents License** for its contents. Check the source licence before any redistribution beyond the course project.
- The source does not give a reliable listing date or region for each row. `selling_price` is the asking price in the listing, not a confirmed sale amount. Therefore the model is an academic estimate for this dataset, not a current market valuation tool.

## Files

- `used_car_price_estimation.py`: the runnable course notebook in Python cell format (`# %%`). It loads data, audits and cleans it, makes four EDA plots, splits the data, trains a baseline and three lecture models, tunes a tree with cross-validation, evaluates the selected model once on the test set, and analyzes errors.
- `car_details_v3.csv`: source dataset for reproducible offline runs.
- `results/`: generated charts and `metrics.json` from the verified run.
- `requirements.txt`: Python package requirements.

## Setup and run

Use Python 3.10 or newer. From this folder:

```powershell
python -m pip install -r requirements.txt
python used_car_price_estimation.py
```

The script writes its figures and metrics to `results/`. It uses random seed 42. It requires the CSV beside the script and does not download data while running.

## Data preparation and evaluation design

We removed 1,202 exact duplicate rows **before** splitting, leaving 6,926 listings. This prevents the same complete record from appearing in training and evaluation. No remaining row failed the basic validity checks for positive price, nonnegative kilometres, plausible year, and nonmissing name. Missing engine, power and seats values are imputed within each training fold. We parse engine capacity in cc and power in bhp from strings, extract brand from the first word of the car name, and use `log1p(km_driven)` to reduce mileage skew. We exclude the inconsistent `torque` strings and mixed-unit `mileage` field at this stage.

The split is 4,848 training, 1,039 validation and 1,039 test rows (70/15/15). Only training rows feed the EDA plots and the three-fold cross-validation. Encoding, scaling and imputation live inside the model pipelines, so each fold learns preprocessing from its own training portion. Validation MAE selects the model. The test set is used once for final reporting. Rare brands can occur only in validation or test; one-hot encoding handles unseen categories.

## Exploratory findings

The training median price is INR 400,000, while the mean is INR 523,009, showing a long expensive tail. Model year correlates positively with price (Spearman rho 0.71), and kilometres driven correlates negatively (rho -0.30). Fuel groups have different observed price distributions, but this does not establish a causal fuel effect. The four plots in `results/` show price distribution, median price by year, mileage versus price, and price by fuel type.

## Current results

Validation MAE, in INR:

| Model | Validation MAE |
| --- | ---: |
| Training-median baseline | 258,331 |
| Linear regression | 133,737 |
| Decision tree regression | 93,186 |
| KNN regression | **90,442** |

The tree used three-fold CV on the training set to choose `max_depth=10` and `min_samples_leaf=3` (CV MAE INR 107,233). KNN had the lowest validation MAE and was selected. On the untouched test set, KNN achieved **MAE INR 98,139**, RMSE INR 205,422 and R² 0.828. The median baseline's test MAE was INR 263,674. KNN therefore improved test MAE by about 62.8% and met the stated midterm criterion. It remained trained on the training partition only; the validation rows were used for model selection.

Error is larger on expensive listings. In the highest training-price quartile, mean absolute test error is INR 203,948, versus INR 47,672 in the lowest quartile. The model may have difficulty with rare high-end brands and trims, and the feature set omits region, condition and listing date. Random splits also do not measure performance on newer listings.

## Team contributions

The team confirmed the following division of technical work.

| Student | Technical contribution |
| --- | --- |
| Alisher Akhmet | Dataset sourcing, data audit, cleaning choices and four EDA findings |
| Aslan Muratov | Model training, cross-validation, evaluation metrics and error analysis |

## Final-stage plan

1. Confirm the original collection period and price definition with better provenance.
2. Add listing date, region, condition and trim if an appropriate licensed dataset becomes available.
3. Compare stronger models and tune them using training-only cross-validation.
4. Evaluate errors by brand and price segment, estimate uncertainty, and use a later-time holdout if listing dates are available.
