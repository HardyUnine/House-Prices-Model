
## 1. Ordinal-Only Model (ANOVA-Based)

**Features Used:**
- `KitchenQual_code`, `GarageFinish_code`, `BsmtQual_code`, `ExterQual_code`, `BsmtExposure_code`
- Interaction terms: `ExterQual × KitchenQual`, `BsmtQual × BsmtExposure`

**Performance:**
- R²: ~0.639
- Pros: Simple, interpretable
- Cons: Missed key numeric features

---

## 2. Full Numerical Model

**Features Used:**
- All numeric variables from `numerical_cleaned.csv`

**Performance:**
- R²: ~0.772
- Cons: Multicollinearity present, large prediction errors on outliers

---

## 3. Improved Numerical Model

**Changes Applied:**
- Dropped redundant feature `TotRmsAbvGrd`
- Removed outliers (residuals > 3 std deviations)

**Performance:**
- R²: ~0.862
- Pros: Cleaner residuals, more stable coefficients

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

---

## Summary Table

| Model Version               | R²     | Notes                                   |
|----------------------------|--------|------------------------------------------|
| Ordinal Only               | ~0.64  | Based on 2ᵏ factorial design & ANOVA     |
| Full Numerical             | ~0.77  | Stronger but noisy                       |
| Improved Numerical         | ~0.86  | Cleaner and more stable                  |
| Log-Transformed (Final)    | **0.89** | Best performance and interpretability  |

