
# House Price Prediction Using ANOVA and Linear Regression

## Overview

This analysis aims to model house prices using categorical and numerical variables from the Ames Housing dataset. The modeling process involves building a series of linear regression models with increasing complexity, starting with one-way ANOVA and progressing to a more comprehensive linear model.

## Data Preparation

The dataset used is the `train.csv` file from the Ames Housing competition. We begin by loading the dataset and cleaning it by removing any rows with missing values in the key variables of interest: `Neighborhood`, `OverallQual`, `GrLivArea`, and `SalePrice`.

```python
df = pd.read_csv('../data/raw/train.csv')
df_clean = df.dropna(subset=['Neighborhood', 'OverallQual', 'GrLivArea', 'SalePrice'])
```

## Feature Correlation Analysis

We compute the absolute Pearson correlation between `SalePrice` and all numeric features to identify the most influential predictors.

```python
numeric_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=['SalePrice', 'Id'])
correlations = numeric_features.corrwith(df['SalePrice']).abs().sort_values(ascending=False)
top_numeric_features = correlations.head(10)
print(top_numeric_features)
```

This helps in identifying strong predictors such as `OverallQual`, `GrLivArea`, and others.

## Model 1: One-Way ANOVA with Neighborhood

We first test whether `Neighborhood` has a statistically significant effect on `SalePrice` using a one-way ANOVA model.

```python
model_one_way = smf.ols('SalePrice ~ C(Neighborhood)', data=df_clean).fit()
anova_one_way = sm.stats.anova_lm(model_one_way, typ=2)
```

**Result**: `Neighborhood` is highly significant (p < 0.001), confirming that location is a strong factor in house pricing.

## Model 2: Two-Way ANOVA with Neighborhood and OverallQual

Next, we incorporate `OverallQual` as a categorical variable and include its interaction with `Neighborhood`.

```python
model_two_way = smf.ols('SalePrice ~ C(Neighborhood) * C(OverallQual)', data=df_clean).fit()
anova_table = sm.stats.anova_lm(model_two_way, typ=2)
```

**Result**: Both main effects (`Neighborhood`, `OverallQual`) and their interaction are statistically significant. This suggests that the impact of house quality varies by neighborhood.

## Model 3: Add GrLivArea as a Continuous Predictor

Finally, we include `GrLivArea` (above ground living area) as a continuous covariate, which converts the model into an ANCOVA-style model.

```python
model_three_way = smf.ols('SalePrice ~ C(Neighborhood) * C(OverallQual) + GrLivArea', data=df_clean).fit()
anova_three_way = sm.stats.anova_lm(model_three_way, typ=2)
```

**Result**: 
- `GrLivArea` is highly significant (p < 0.001).
- The adjusted R-squared of this model is approximately 0.864, indicating it explains about 86% of the variance in sale prices.
- This model balances interpretability and predictive power effectively.

## Conclusion

We built and compared three models:

| Model          | Factors Included                               | Adjusted R² | Notes                                      |
|----------------|------------------------------------------------|-------------|--------------------------------------------|
| Model 1        | `Neighborhood`                                 | ~0.66       | Basic spatial effects                      |
| Model 2        | `Neighborhood`, `OverallQual`, interaction     | ~0.801       | Interaction improves explanatory power     |
| Model 3 (final)| Above + `GrLivArea`+ `OverallQual_Cat` removed | **0.864**   | Best balance of performance and simplicity |

We selected **Model 3** as the final model due to its superior explanatory power and inclusion of a continuous, interpretable variable (`GrLivArea`) that is highly correlated with price.

