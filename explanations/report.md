# 1. Problem statement 

## Problem Statement

### Objective

The goal of this project is to apply core statistical modeling techniques to a real-world dataset in a collaborative setting, simulating the work of a data science team. Specifically, we aim to analyze and predict housing prices using a structured and statistically grounded approach.

The dataset, provided by Kaggle, contains rich housing market data with numerous features:  
[House Prices: Advanced Regression Techniques – Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

### Group Work

 The key objectives include:

- Data preprocessing
- Statistical model development
- Prediction of housing prices
---

### Required Methodologies

Throughout this project, we tried to apply the following statistical techniques:

1. **Classical Statistical Inference**
   - Sample mean and variance
   - Confidence intervals
   - Hypothesis testing

2. **Design of Experiments and ANOVA**
   - 2^k factorial design
   - Fractional factorial design
   - Analysis of variance (ANOVA)

3. **Regression Modeling and ANOVA**
   - Build predictive regression models
   - Use ANOVA to interpret model fit and significance of predictors

4. **Time Series Analysis**
   - Extract and process house price time series data
   - Apply SARIMA/SARIMAX models
   - Use log transformation, differencing, decomposition, and exogenous features for forecasting

# 2. Data Cleaning and Preparation

## Overview

This section covers the essential preprocessing steps applied to both the training and test datasets before modeling. The focus is on handling missing data, transforming categorical variables, and preparing a clean dataset suitable for the next steps.

This notebook was largely inspired by the Data Exploration, Engineering and Cleaning section of abhinand5's take on the housing prices kaggle competition
 
[predicting-housingprices-simple-approach](https://github.com/abhinand5/Housing-Prices-Advanced-Regression-Techniques--KAGGLE-CHALLENGE/blob/master/predicting-housingprices-simple-approach-lb-top3.ipynb)
---

## Importing Necessary Packages

The following Python libraries were used in this process:

- **NumPy** (`numpy`) – for efficient numerical operations and array manipulation.
- **Pandas** (`pandas`) – for reading, processing, and analyzing tabular data (e.g., CSV files).
- **Matplotlib** (`matplotlib.pyplot`) and **Seaborn** (`seaborn`) – for creating informative visualizations.
- **Scikit-learn** (`sklearn`) – for preprocessing tasks such as encoding categorical variables.

---

## Loading the Data

```python
train = pd.read_csv('../data/raw/train.csv', index_col='Id')
test = pd.read_csv('../data/raw/test.csv', index_col='Id')
```

Both the training and test sets were loaded, with `Id` set as the index.

---

## Exploring Missing Data

We created a bar chart to show which columns contain missing values.

```python
def plot_missing(df):
    ...
plot_missing(train)
```
---

## Imputing Missing Values

A function `fill_missing_values()` was defined to handle missing data:

- **Categorical columns** were filled with their **most frequent value**
- **Numeric columns** were filled with their **median**

```python
fill_missing_values(train)
fill_missing_values(test)
```

This method has a few problems, the biggest one being that it replaces the missing data by the most frequent value. Meaning that it can change a lot the distribution of values in a variable.

---

## Encoding Categorical Variables

Machine learning models require numerical input. A function `impute_cats()` was created to encode string-based categorical variables using `LabelEncoder`.

```python
impute_cats(train)
impute_cats(test)
```

This final step transforms all non-numeric columns into usable numeric representations for modeling.

---

## Conclusion

The resulting training and test datasets are now free of missing values and fully numerical.
# 3. 2k Factorial Design

##########################################

# 4. Analysis trough statistical inference
## Data Preparation

We selected only **purely numeric variables**, excluding ordinal-encoded or categorical-like numeric variables such as `MSSubClass`, `OverallQual`, and `MoSold`, to ensure the appropriateness of the statistical techniques applied.

```python
numeric_df = train.select_dtypes(include=[np.number]).copy()
numerical_df = numeric_df.drop(columns=["MSSubClass", "OverallQual", "OverallCond", "MoSold"])
```

---

## Descriptive Statistics

We first computed basic summary statistics for each numeric variable, including:

- Mean and standard deviation
- Minimum and maximum values
- Quartiles (25%, 50%, 75%)

These summaries help describe the central tendency and spread of the data.

---

## Distribution and Normality Checks

To assess whether the distribution of `SalePrice` follows a normal distribution, we used:

- **Histogram with KDE (Kernel Density Estimate)**
- **QQ Plot (Quantile-Quantile plot)**

The original `SalePrice` distribution showed a **positive skew**, with high-value outliers creating a long right tail. To correct this, we applied a **logarithmic transformation**:

```python
numerical_df['LogSalePrice'] = np.log(numerical_df['SalePrice'])
```

The QQ plot of the log-transformed prices showed a more linear pattern, indicating a **closer approximation to normality**.

---

## Confidence Interval for the Mean Sale Price

We computed a **95% confidence interval** for the population mean of `SalePrice` using the **t-distribution**:

```python
conf_int = stats.t.interval(0.95, df=n-1, loc=mean, scale=std/np.sqrt(n))
```

This gives a range in which the true mean sale price is expected to fall with 95% confidence.

---

## Hypothesis Testing

We tested the hypothesis:

- **Null hypothesis (H₀):** mean = \$180,000  
- **Alternative hypothesis (H₁):** mean > \$180,000  

This was conducted using a one-sample t-test:

```python
t_stat, p_value = stats.ttest_1samp(numerical_df['SalePrice'], popmean=180000)
```

The test yielded a significant result, indicating that the average `SalePrice` in the sample is **statistically significantly greater** than \$180,000.

# 5. Analysis trough ANOVA/DOE

## Overview

This analysis explores the determinants of house prices in the Ames Housing dataset using a progression of increasingly complex statistical models. The final model was what we used to generate sale price predictions for the test dataset submitted to the Kaggle competition.

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
| Model 2        | `Neighborhood × OverallQual`                     | Captures interaction effects (Adj R² ≈ 0.784) |
| Model 3 (final)| `Neighborhood × OverallQual` `+` `GrLivArea`     | Best overall performance (Adj R² ≈ 0.852) |

## Kaggle results
The Kaggle House Prices competition evaluates predictions using the Root Mean Squared Logarithmic Error (RMSLE). 
\[
\text{RMSLE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2 }
\]

**Our Result:**
Our final model achieved a RMSLE of 0.18189, which implies that the predictions differ from actual prices by roughly 18% on a relative (logarithmic) scale.

While this performance placed our submission at 3761st out of 4648 competitors, it was achieved using a clean, interpretable linear model with no advanced feature engineering or machine learning, making it a strong baseline and learning outcome.

# V. Analysis Through Time Series

After lot of tries we found that the last value of `Monthly Average SalePrice` was not relevant (only one house) and influenced too much our predictions, we decided to remove it. 

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