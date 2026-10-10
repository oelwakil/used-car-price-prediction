# Used Car Price Prediction (Cars4U)

Predicting used car prices in the Indian market with regression and boosted tree models, and explaining the predictions with SHAP. The final tuned XGBoost model reaches **R² ≈ 0.97 (log price) and ~6% MAPE** on the test split. Because hyperparameter tuning reused that split, treat these figures as optimistic (see [Limitations](#limitations)); the untuned CatBoost baseline (R² 0.969, 6.4% MAPE) is a cleaner estimate.

> Capstone project for the MIT Applied AI and Data Science Professional Program. Notebook: [`Capstone_Project_Reference_Notebook_Full_Code_Used_Cars_Price_Prediction.ipynb`](./Capstone_Project_Reference_Notebook_Full_Code_Used_Cars_Price_Prediction.ipynb)

## Business problem

Cars4U is a used car platform operating in a fragmented, volatile market. Overpriced cars sit on the lot and slow inventory turnover. Underpriced cars leave revenue on the table. The goal was a data-driven model that estimates a used car's market price from its listing attributes, and that can say *why* it priced a car the way it did.

This is a supervised regression problem. The target is the used car price (INR 100,000 units), modeled on a log scale.

## Data

- **Source:** a used car listings dataset (`used_cars.csv`) provided by the MIT AAIDS program for this capstone, 7,253 listings and 14 columns. The dataset is not included in this repository.
- **Features:** name (brand and model), location (11 cities), year, kilometers driven, fuel type, transmission, owner type, mileage, engine (CC), power (bhp), seats, and the new-car price of the same model.
- **Target:** `Price`, missing for 1,234 listings. Those rows were dropped, since imputing the target would bias the model.
- **Main data challenge:** `New_price` was missing for ~86% of listings, yet it turned out to be the strongest predictor (see below).

## Approach

**1. Cleaning and imputation**
- Dropped rows with a missing target, a single extreme kilometers outlier, and 36 rows with zero mileage.
- Imputed missing `Power` from grouped medians (brand, year, fuel, transmission, engine). Dropped the few rows that could not be imputed.
- Imputed `New_price` in two phases, a granular match first and a broader one second. This cut missingness from 86.1% to 55.3%. Rows still missing it were excluded from the "with new price" models, leaving **2,644** listings versus 5,897 without it.

**2. EDA**
- Univariate and bivariate analysis, correlation heatmap, and box plots of price against location, fuel, transmission, owner type, and brand.
- Log-transformed the skewed variables (price, new price, kilometers, engine, power).

**3. Feature engineering**
- `Car_Age`, `Car_Age_Squared`, `Kilometers_per_Year`, and a premium new-price indicator.
- Interaction terms: kilometers × age, mileage × age, new price × age.
- PCA to combine `log_Power` and `log_Engine` into one component, reducing multicollinearity.
- Brand extracted from the name. Fuel type, owner type, and seats grouped or one-hot encoded.

**4. Modeling.** I built 18 feature-set variants (log vs. original scale, with/without new price, with/without engineered features, with/without PCA) to measure what each choice contributed, then fit:
- **Linear:** OLS with K-fold CV and bootstrapped confidence intervals, robust standard errors for heteroskedasticity (Breusch-Pagan), VIF checks, Ridge, and Lasso.
- **Tree-based:** Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost.
- **Tuning:** GridSearchCV, then Bayesian optimization with Optuna (100 and 200 trials) on the two strongest boosting models.
- **Interpretation:** SHAP summary, bar, dependence, and force plots for the final model.

All models use an 80/20 train/test split with `random_state=42`.

## Results

Test set, 80/20 split. R², RMSE, and MAE are on log price. MAPE is the percentage error.

| Model | Test R² | Test RMSE | Test MAE | Test MAPE |
|---|---|---|---|---|
| OLS, robust SEs (best linear) | 0.9528 | 0.1751 | 0.1343 | 8.15% |
| CatBoost (untuned) | 0.9688 | 0.1423 | 0.1035 | 6.41% |
| XGBoost (untuned) | 0.9630 | 0.1550 | 0.1099 | 6.89% |
| **XGBoost (Optuna, 200 trials)** | **0.9709** | **0.1375** | **0.0997** | **6.10%** |

Without the new-price feature, the best tree model (CatBoost) reaches R² 0.9662 but MAPE rises to 11.7%. That comparison is on the larger 5,897-listing sample, so it is indicative rather than like-for-like.

### Key findings

- **New car price dominates.** `log_New_price` is the most influential predictor in both the linear models and SHAP, followed by the engine power/size component. Models with it have markedly lower percentage error.
- **Depreciation is an interaction effect.** Age, usage, and original price combine. High mileage on an older car lowers value more than either factor alone.
- **Market effects are real.** Brand (premium vs. economy), location (e.g., Kolkata sits below the baseline city), transmission, fuel type, and owner history all shift price.
- **Boosting beats linear models** at capturing these non-linearities. Robust OLS stays useful where coefficient-level interpretation matters, and Ridge and Lasso matched its accuracy on this feature set.
- **Heteroskedasticity persisted** in the OLS model even after robust standard errors, which fix inference but not the underlying variance pattern. One-hot brand and seat dummies produced very high VIFs.

### Recommendations from the analysis

1. Invest in collecting complete new-car price data, since it is the highest-value feature.
2. Deploy the tuned model behind a monitored pipeline, with drift tracking and periodic retraining.
3. Give pricing teams SHAP explanations so predictions can be audited and discussed.
4. Consider segment-specific models (luxury, electric) and external market indicators.

## Limitations

- **Tuning and test data overlap.** The Optuna objective scored candidates on a validation split created with the same `test_size=0.2, random_state=42` as the final evaluation split. The tuned models' test metrics are therefore optimistic. The untuned CatBoost result (R² 0.9688) is a cleaner estimate, and a proper fix is a three-way split or nested cross-validation.
- **Imputed predictor.** More than half of `New_price` is missing, and the models using it train on 2,644 listings, many with imputed values.
- **No time dimension.** The data is a snapshot, so the model does not capture seasonality or market shifts.
- **Tree models were evaluated on a single split** (no CV or bootstrapping), unlike the linear models, because of compute limits.

## Repository contents

```
.
├── Capstone_Project_Reference_Notebook_Full_Code_Used_Cars_Price_Prediction.ipynb
└── README.md
```

## Reproducing the results

The notebook was developed in Google Colab. It mounts Google Drive and reads `/content/used_cars.csv`, so update the path (and remove the Drive-mount cell) to run it elsewhere.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels xgboost lightgbm catboost optuna shap
jupyter notebook
```

The dataset is not included because it was provided by the program. Place your own copy of `used_cars.csv` where the notebook expects it and run the cells top to bottom. Optuna tuning (100 and 200 trials per model) is the slowest step.

## Tech stack

Python, pandas, NumPy, scikit-learn, statsmodels, XGBoost, LightGBM, CatBoost, Optuna, SHAP, matplotlib, seaborn.
