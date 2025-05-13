## 2ᵏ Factorial Design and ANOVA — Summary
In this step, we applied a 2ᵏ factorial design combined with Analysis of Variance (ANOVA) to identify the most important ordinal features influencing house prices.

We began by selecting 6 ordinal or binary variables with low cardinality and strong correlation with SalePrice. These were:

KitchenQual_code

GarageFinish_code

BsmtQual_code

ExterQual_code

BsmtExposure_code

HeatingQC_code

Using these, we built a factorial model including all main effects and two-way interactions. We then performed an ANOVA to evaluate which variables and combinations had a statistically significant impact on SalePrice.

## Key Results
All main effects (except HeatingQC_code) were highly significant with p-values near zero, indicating a strong relationship with house prices.

HeatingQC_code showed no significance and can be removed from further analysis.

Two interaction effects — ExterQual × KitchenQual and BsmtQual × BsmtExposure — were also statistically significant, suggesting that the impact of one feature depends on the value of the other.

Most other interaction terms were not significant and can be ignored in simplified models.

## Conclusion
The ANOVA analysis helped us identify 5 important predictors of SalePrice and 2 relevant interaction effects. These findings will guide feature selection and model structure in the next step: regression analysis.

