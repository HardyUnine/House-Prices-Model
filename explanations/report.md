# Ceci est le fichier du rapport
je pense que le mieux c'est de se baser sur les explications de chaque numéro pour ensuite le compléter 
# I Problem statement 

The goal of our project 

# II Data clearing (:clownface:)

# III Analysis trough statistical inference

# IV Analysis trough ANOVA/DOE

## Overview

This analysis explores the determinants of house prices in the Ames Housing dataset using a progression of increasingly complex statistical models. The objective was to identify and validate the most predictive features using ANOVA and ANCOVA methodologies, while maintaining interpretability. The final model was deployed to generate sale price predictions for the test dataset submitted to the Kaggle competition.

## What is ANCOVA?
ANCOVA (Analysis of Covariance) is a statistical method that combines ANOVA and linear regression. It allows you to examine the effect of categorical variables (factors) on a continuous outcome while controlling for one or more continuous variables (covariates).

## Data Preparation

The raw data was loaded from the training set, and rows with missing values in essential variables (`Neighborhood`, `OverallQual`, `GrLivArea`, `SalePrice`) were removed to ensure clean modeling inputs.

```python
df = pd.read_csv('../data/raw/train.csv')
df_clean = df.dropna(subset=['Neighborhood','OverallQual','GrLivArea', 'SalePrice'])
```

## Correlation Analysis

To identify the most influential predictors of `SalePrice`, we calculated the absolute correlation of all numeric variables with the target variable. The top 10 features were examined, with `OverallQual` and `GrLivArea` showing the strongest positive correlation with price.

```python
numeric_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=['SalePrice', 'Id'])
correlations = numeric_features.corrwith(df['SalePrice']).abs().sort_values(ascending=False)
top_numeric_features = correlations.head(10)
```

## Model 1: One-Way ANOVA (Neighborhood Only)

The first model tested whether average house prices varied significantly by `Neighborhood`. A one-way ANOVA was performed using `Neighborhood` as a categorical predictor.

```python
model_one_way = smf.ols('SalePrice ~ C(Neighborhood)', data=df_clean).fit()
anova_one_way = sm.stats.anova_lm(model_one_way, typ=2)
```

**Findings**:  
The model showed that `Neighborhood` is a highly significant factor in determining house prices. The adjusted R² confirmed the strength of this spatial component.

## Model 2: Two-Way ANOVA (Neighborhood × OverallQual)

We then added `OverallQual` as a categorical variable and included an interaction term with `Neighborhood` to account for differential effects of house quality across locations.

```python
model_two_way = smf.ols('SalePrice ~ C(Neighborhood) * C(OverallQual)', data=df_clean).fit()
anova_table = sm.stats.anova_lm(model_two_way, typ=2)
```

**Findings**:  
Both main effects and their interaction were significant, indicating that the influence of quality depends on the neighborhood context.

## Model 3: ANCOVA with GrLivArea

To improve the model's predictive power while maintaining interpretability, we added `GrLivArea` as a continuous covariate. This turned the model into an ANCOVA, combining both categorical and numeric predictors.

```python
model_three_way = smf.ols('SalePrice ~ C(Neighborhood) * C(OverallQual) + GrLivArea', data=df_clean).fit()
anova_three_way = sm.stats.anova_lm(model_three_way, typ=2)
```

**Findings**:  
The inclusion of `GrLivArea` significantly improved the model fit, raising the adjusted R² to approximately 0.789. This model balances statistical significance with real-world interpretability and was selected as the final model.

## Prediction and Submission

Using the final model, we predicted house prices on the test dataset. Only records with complete data were retained. Predictions were saved to `submission.csv` for Kaggle evaluation.

```python
df_test = pd.read_csv("../data/raw/test.csv")  
df_test = df_test.dropna(subset=['Neighborhood', 'OverallQual', 'GrLivArea'])
df_test['SalePrice'] = model_three_way.predict(df_test)
df_test[['Id', 'SalePrice']].to_csv("submission.csv", index=False)
```

## Conclusion

Three models were compared through ANOVA and regression metrics:

| Model          | Predictors                                        | Notes                               |
|----------------|--------------------------------------------------|--------------------------------------|
| Model 1        | `Neighborhood`                                   | Establishes spatial effect (Adj R² ≈ 0.538) |
| Model 2        | `Neighborhood × OverallQual`                     | Captures interaction effects (Adj R² ≈ 0.760) |
| Model 3 (final)| `Neighborhood × OverallQual` `+` `GrLivArea`     | Best overall performance (Adj R² ≈ 0.852) |

## Kaggle results
The Kaggle House Prices competition evaluates predictions using the Root Mean Squared Logarithmic Error (RMSLE). 
\[
\text{RMSLE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2 }
\]

**Our Result:**
Our final model achieved a RMSLE of 0.18189, which implies that the predictions differ from actual prices by roughly 18% on a relative (logarithmic) scale.

While this performance placed our submission at 3761st out of 4648 competitors, it was achieved using a clean, interpretable linear model with no advanced feature engineering or machine learning, making it a strong baseline and learning outcome.

## V. Analysis Through Time Series (Deep Dive)

To begin our time series analysis, we first inspect the dataset to understand the completeness and quality of each variable. This is particularly important for modeling with exogenous variables. For example, features like `Alley` have very few non-null values (e.g., only 91 entries), indicating they are not reliable for modeling and may be excluded or imputed.

We then explore the evolution of `Monthly Average SalePrice` over time. Initially, the raw time series plot does not reveal clear patterns. However, upon differencing the series (i.e., calculating month-over-month changes), we observe notable fluctuations but no stable pattern—hinting at possible non-stationarity.

To better understand the components of the time series, we use **seasonal decomposition** from `statsmodels`, which splits the series into three parts:
- **Trend**: Shows a declining trajectory, dropping from about \$185,000 to below \$175,000 over the study period.
- **Seasonal**: Exhibits a clear annual pattern, with price peaks and valleys repeating regularly every 12 months.
- **Residual**: Captures remaining variation after removing trend and seasonality.

This decomposition supports the use of **seasonal modeling** with a period of 12 months.

### Trend Significance: Stationarity Check

To statistically validate the presence (or absence) of a trend, we fit linear models to the time index and compute the **p-value** of the slope (β₁):

- **Original Series**:  
  - `p = 0.0254`  
  - Suggests a statistically significant trend → the series is **non-stationary**.

- **Log-Differenced Series**:  
  - `p = 0.5835`  
  - Trend is no longer significant → the transformation has made the series **stationary**.

This confirms that log transformation + first differencing effectively removes the trend, justifying `d = 1` in our SARIMA model.

### ACF and PACF Analysis

To choose appropriate SARIMA orders, we examined:

- **ACF (Autocorrelation Function)**:  
  Strong spike at lag 1, with decay over 12 lags — indicates short- and medium-term memory.

- **PACF (Partial Autocorrelation Function)**:  
  Significant at lag 1 and slightly beyond — supports using AR terms (p = 1 or 2).

This analysis supports trying models like `(p,d,q) = (1,1,1)` and seasonal `(P,D,Q,12) = (1,1,1,12)`.

### SARIMA Model Variants

We tested four SARIMA-based models:
1. **Basic SARIMA** using `Monthly Average SalePrice`.
2. **SARIMA with log-transformed target** using `Monthly Average SalePrice`.
3. **SARIMAX with exogenous variables** (`YearBuilt`, `OverallQual`) using log target.
4. **SARIMAX with exogenous variables** (`YearBuilt`, `OverallQual`) using raw target (non-log).

Models were evaluated using AIC and variance of relative forecast error, with log-transformed models performing more robustly.

---

## Forecasting Future Values

We used the final SARIMAX model to forecast the next **6 months** of average sale prices.

- Used the most recent average of `OverallQual` for future exogenous input.
- Forecasted in **log scale**, then exponentiated for interpretability.
- Results show smooth projected continuation aligned with historical trend and seasonality.

---

## Kaggle Submission 

The SARIMAX model was applied to the Kaggle test set to generate predictions:

- Trained using `(3, 1, 3)` with seasonal `(1, 0, 1, 12)`, and exogenous variables.
- Aggregated test data by month and computed average `YearBuilt` and `OverallQual`.
- Forecasted log prices → exponentiated → merged predictions back to test records.
- Saved output as `submission.csv` with `Id` and `SalePrice`.

This pipeline aligns with real-world deployment practices and ensures robust, seasonally-aware forecasting.



## VI Our inexistent novel method