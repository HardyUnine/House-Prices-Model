## 1. Ordinal-Only Model (ANOVA-Based)

**Features Used:**
- `KitchenQual_code`, `GarageFinish_code`, `BsmtQual_code`, `ExterQual_code`, `BsmtExposure_code`
- Interaction terms: `ExterQual × KitchenQual`, `BsmtQual × BsmtExposure`

**Performance:**
- R²: ~0.639
- Pros: Simple, interpretable
- Cons: Missed key numeric features

**Plot Insights:**
- Actual vs Predicted shows large variance and inconsistent accuracy.
- Residuals vs Fitted displays a fan shape → increasing variance with predicted value.
- Histogram shows mild skewness and wide error distribution.
- QQ plot reveals non-normality at tails — residuals deviate from the expected diagonal line.

---

## 2. Full Numerical Model

**Features Used:**
- All numeric variables from `numerical_cleaned.csv`

**Performance:**
- R²: ~0.772
- Cons: Multicollinearity present, large prediction errors on outliers

**Plot Insights:**
- Residuals show large spread and non-normality.
- QQ plot deviates significantly from diagonal.
- Histogram shows skewed errors and long tails.

---

## 3. Improved Numerical Model

**Changes Applied:**
- Dropped redundant feature `TotRmsAbvGrd`
- Removed outliers (residuals > 3 std deviations)

**Performance:**
- R²: ~0.862
- Pros: Cleaner residuals, more stable coefficients

**Plot Insights:**
- Residuals are better centered and spread is narrower.
- Histogram is more symmetric.
- QQ plot shows moderate alignment with the diagonal.

---

## 4. Final Model: Log-Transformed SalePrice

**Changes Applied:**
- Log-transformed `SalePrice`
- Refit model after outlier removal

**Performance:**
- R²: 0.894
- Pros:
  - Best model so far
  - Residuals are well-behaved
  - Coefficients interpretable as percentage effects

**Plot Insights:**
- Residuals on log scale closely follow a normal distribution.
- QQ plot using log residuals shows strong linearity.

---

## Summary Table

| Model Version               | R²     | Notes                                   |
|----------------------------|--------|------------------------------------------|
| Ordinal Only               | ~0.64  | Based on 2ᵏ factorial design & ANOVA     |
| Full Numerical             | ~0.77  | Stronger but noisy                       |
| Improved Numerical         | ~0.86  | Cleaner and more stable                  |
| Log-Transformed (Final)    | **0.89** | Best performance                       |