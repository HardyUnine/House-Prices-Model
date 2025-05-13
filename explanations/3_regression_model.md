###  Step 3: Regression and ANOVA Analysis

We fit a linear regression model to predict `SalePrice` using key ordinal features and significant interaction terms identified from Step 2.

**Main Effects Used:**
- `KitchenQual_code`
- `GarageFinish_code`
- `BsmtQual_code`
- `ExterQual_code`
- `BsmtExposure_code`

**Interaction Terms:**
- `ExterQual_code × KitchenQual_code`
- `BsmtQual_code × BsmtExposure_code`

---

####  Model Performance

| Metric               | Value       |
|----------------------|-------------|
| R-squared            | 0.639       |
| Adjusted R-squared   | 0.637       |
| F-statistic          | 366.7       |
| p-value (model)      | 1.38e-315    |
| Observations         | 1460        |

The model explains ~64% of the variation in `SalePrice` and is statistically highly significant.

---

####  ANOVA Model Comparison

| Model               | SSR          | df    | F       | p-value       |
|---------------------|--------------|-------|---------|---------------|
| Main effects only   | 3.67e+12     | 1454  | –       | –             |
| Full (w/ interactions) | 3.33e+12  | 1452  | 75.67   | 5.50e-32      |

Adding interaction terms significantly improves the model fit (**p ≪ 0.001**).

---

####  Key Coefficients

- `KitchenQual_code`: -55,610  
- `GarageFinish_code`: +14,570  
- `ExterQual_code × KitchenQual_code`: +25,790  
- `BsmtQual_code × BsmtExposure_code`: +9,999  

Interaction terms add nuance: for example, better kitchen and exterior quality together lead to significantly higher prices.

---

####  Residuals Summary

- Mean: ≈ 0  
- Std dev: ~47,750  
- Range: from -259k to +407k  
→ Some under/overestimation expected in housing datasets

---