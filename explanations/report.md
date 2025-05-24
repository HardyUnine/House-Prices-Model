# Ceci est le fichier du rapport
je pense que le mieux c'est de se baser sur les explications de chaque numéro pour ensuite le compléter 
## I Problem statement 

## II Data clearing (:clownface:)

## III Analysis trough statistical inference

## IV Analysis trough ANOVA/DOE

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
2. **SARIMA with log-transformed target**.
3. **SARIMAX with exogenous variables** (`YearBuilt`, `OverallQual`) using log target.
4. **SARIMAX with exogenous variables** using raw target (non-log).

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