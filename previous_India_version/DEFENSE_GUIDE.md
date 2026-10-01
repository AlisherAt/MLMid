# Midterm defense guide (10 minutes)

**Project:** Used car price estimation  
**Presenters:** Алишер Ахмет (Alisher Akhmet) and Муратов Аслан (Aslan Muratov)

This is a speaking guide for the submitted 10-slide deck. The team confirmed the division: Alisher covers the dataset, cleaning and EDA; Aslan covers modelling and evaluation. Both presenters should be able to explain the entire project.

| Slide | Speaker | Time | What to say |
| --- | --- | ---: | --- |
| 1. Title | Alisher | 0:35 | “We estimate advertised prices of used cars. We use a real CarDekho listing dataset from India and report errors in Indian rupees.” |
| 2. Question | Alisher | 1:00 | “This is regression. Our target is `selling_price`, the asking price. MAE is our main metric because it tells us the average absolute error in rupees. We set a midterm criterion of MAE below INR 150,000 and at least 20% below the median-price baseline.” |
| 3. Dataset | Alisher | 1:05 | “The public dataset has 8,128 records and 13 columns. The model years range from 1983 to 2020. The source lacks listing dates and locations, so the model is an academic result on historical listings, not a current market appraisal.” |
| 4. Year EDA | Alisher | 1:05 | “We made all four EDA plots using the training partition. The median asking price generally rises with model year. Spearman correlation is 0.71. This association does not prove that year alone causes the difference, because brands and specifications vary too.” |
| 5. Fuel EDA | Alisher | 0:50 | “Fuel groups have different price distributions. Diesel has the highest median in these data, but group composition may explain part of the difference. The training mean exceeds the median because expensive cars form a long right tail.” |
| 6. Preparation | Alisher | 1:05 | “We removed 1,202 exact duplicate rows before splitting. We parsed engine capacity and power from strings, extracted brand from name, and used log kilometres. Imputation, encoding and scaling happen inside each model pipeline, so each training fold learns preprocessing without evaluation data.” |
| 7. Evaluation | Aslan | 1:00 | “The split is 4,848 training, 1,039 validation and 1,039 test rows. We compare a training-median baseline with linear regression, decision tree and KNN. Three-fold cross-validation on training data chose the tree’s depth and minimum leaf size.” |
| 8. Validation | Aslan | 1:15 | “KNN had the lowest validation MAE, about INR 90,442, slightly ahead of the tree at INR 93,186. We selected KNN using validation data before examining the held-out test set.” |
| 9. Test | Aslan | 1:25 | “KNN achieved test MAE INR 98,139, RMSE INR 205,422 and R² 0.828. The median baseline test MAE was INR 263,674, so our MAE improved by about 63%. Error is much larger in the most expensive price quarter, with mean absolute error about INR 203,948.” |
| 10. Next steps | Aslan | 0:40 | “For the final stage, we need better source provenance, dated and regional listing fields if licensed, stronger models and subgroup checks. A later-time test would better measure performance on future listings.” |

**Total: 10:00.** Practice once with a timer. The figures are rounded for speech; exact values are in `results/metrics.json` and `README.md`.

## Likely questions and concise answers

**What exactly does the model predict?** The price written in a listing (`selling_price`), in INR. The dataset does not verify final transaction prices.

**Why remove duplicate rows before splitting?** Otherwise identical records could appear on both sides of the split and make evaluation too optimistic.

**How do you avoid preprocessing leakage?** Missing-value imputation, scaling and category encoding are fitted inside scikit-learn pipelines. During cross-validation, each fold fits them using only that fold’s training rows.

**Why have separate validation and test sets?** Validation chooses the model. The held-out test provides a final estimate after model choice.

**Why use MAE?** It measures the average size of an error in the same currency as the price and is easier to explain than squared error. RMSE and R² provide additional context.

**Why did KNN win?** It had the lowest validation MAE among the tested models. We did not choose it using the test result.

**Why is the error higher for expensive cars?** High-end listings are rarer, diverse in trim and condition, and some important attributes are missing. The error-by-price-band chart supports the pattern, but the exact cause needs further analysis.

**Can this estimate a car’s price today?** The dataset is historical and lacks listing dates and regions. The reported score applies to a random held-out sample of this dataset, not a current live market.

**What did each person do?** Alisher handled dataset sourcing, auditing, cleaning and EDA. Aslan handled model training, cross-validation, metrics and error analysis. Both explain their sections in the defense.
