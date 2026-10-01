# Kazakhstan used-car project: 10-minute defense

**Presenters:** Алишер Ахмет (Alisher Akhmet) and Муратов Аслан (Aslan Muratov). The team confirmed that Alisher covers data and EDA; Aslan covers modelling and evaluation. Both should know the whole workflow.

1. **Title — Alisher, 0:35.** “We estimate used-car asking prices in Kazakhstan in tenge, using real Kolesa.kz advertisements distributed as a public Kaggle dataset.”
2. **Question — Alisher, 1:00.** “This is regression. The target is the asking price in KZT. MAE tells us the average absolute error in the same currency. We aim to beat a training-median baseline by at least 20% on held-out test MAE.”
3. **Dataset — Alisher, 1:05.** “The dataset was last updated in December 2025. It has 3,999 rows and 14 columns. Prices are labelled in tenge, but the dataset does not give listing dates or cities, so we do not call it live 2026 market data.”
4. **Preparation — Alisher, 1:05.** “We exclude 527 new-car rows and 51 on-order used rows, then remove repeated listings by ad ID before splitting. We have 2,553 unique used-car listings. We recover engine size from descriptions and leave missing mileage for fold-fitted imputation.”
5. **Year EDA — Alisher, 1:05.** “On training data, newer model years generally list for more. Spearman rho is 0.69. This is association, not proof that year alone causes the price difference.”
6. **Fuel and price EDA — Alisher, 0:50.** “The median price is ₸6.70 million and the mean is ₸9.53 million because expensive cars create a long tail. Fuel groups differ, but hybrid and gas have only 22 and 24 training cars, so those medians need caution.”
7. **Evaluation — Aslan, 1:00.** “The split is 1,787 training, 383 validation and 383 test listings. Pipelines fit imputation and encoding within the folds. The decision tree is tuned using three-fold CV on training only. The test set remains untouched during model choice.”
8. **Validation — Aslan, 1:15.** “We compare the median baseline, linear regression, decision tree and KNN. KNN has the lowest validation MAE, ₸3.18 million, and is selected before the final test.”
9. **Test and errors — Aslan, 1:25.** “KNN test MAE is ₸2.445 million, compared with ₸5.951 million for the baseline. That is a 58.9% reduction. Test R² is 0.715. Error is greatest for the most expensive price quarter: ₸6.44 million mean absolute error.”
10. **Limits and next stage — Aslan, 0:40.** “The data are Kazakhstan-specific, yet they are a 2025 snapshot with asking prices and no city or listing date. For the final stage, we need dated 2026 data, stronger models and evaluation on later listings.”

**Total: 10:00.** Exact model values are in `results/metrics.json`. Practise once with a timer.

## Likely questions

**Are these prices actually in tenge?** Yes. Every source `price_raw` label includes ₸, and the numeric `price` column is used directly with no currency conversion.

**Why not convert the earlier Indian dataset to tenge?** Currency conversion would leave the underlying Indian market unchanged. This project uses listings from Kazakhstan instead.

**Are these current 2026 prices?** No. Kaggle last updated this dataset on 18 December 2025, and row-level listing dates are absent. We make no claim about live 2026 pricing.

**What does the model predict?** Advertised asking price, not the amount paid after negotiation.

**Why remove repeated ad IDs before splitting?** The same listing appearing in training and test would make test error look artificially low.

**How do you avoid preprocessing leakage?** Imputation, scaling and one-hot encoding are fitted in the scikit-learn pipeline using only each training fold.

**Why use validation and test separately?** Validation chooses among the lecture models. The test set estimates final performance after that choice.

**Why use MAE?** It expresses average absolute error in tenge and is easier to explain than squared error. We also report RMSE and R².

**Why is error larger on expensive cars?** Those cars are rarer and diverse, while location, condition and trim detail are limited. This is a hypothesis supported by the error pattern, not a proven cause.

**What did each student do?** Alisher handled source data, audit, cleaning and EDA. Aslan handled model training, CV, metrics and error analysis. Both explain their parts and answer questions.
