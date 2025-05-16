# Classical Statistical Inference

In this section, we perform classical statistical inference on **all numerical features** of the housing dataset, with a particular focus on understanding central tendency, spread, confidence intervals, and significance testing.

---

### Descriptive Statistics

We compute basic summary statistics — mean, standard deviation, minimum, quartiles, and maximum — for each numerical variable. This provides a first look at the scale, range, and variability of housing-related features such as `GrLivArea`, `LotArea`, `GarageArea`, and `SalePrice`.

---

### Visualization

We plot:
- A histogram of `SalePrice` to visualize its distribution.
- A Q-Q plot to assess how closely `SalePrice` follows a normal distribution.

These visual tools help evaluate the appropriateness of parametric tests such as t-tests and confidence intervals.

---

### Confidence Interval for SalePrice

We calculate a **95% confidence interval** for the mean of `SalePrice` using the t-distribution:

**180,642.21 to 190,370.10**

This means we are 95% confident that the true population mean of house prices falls within this range.

---

### Hypothesis Testing (1-Sample t-tests)

We test the null hypothesis that each feature has a mean of **zero**, using two-tailed t-tests. While zero is not always a meaningful benchmark for every variable, this process demonstrates how each feature deviates significantly from zero.

- For all features, **p-values are well below 0.05**, indicating that their means are **statistically significantly different from 0**.
- For example:
  - `LotArea`: t = 41.69, p < 0.0001
  - `GrLivArea`: t = 97.90, p < 0.0001
  - `GarageCars`: t = 96.13, p < 0.0001

This confirms that none of the features have a mean near zero, as expected in real-world housing data.

---

### Confidence Intervals for All Features

We also compute 95% confidence intervals for the **mean of every numerical feature**. A few examples:

| Feature        | 95% CI Lower | 95% CI Upper |
|----------------|--------------|--------------|
| `GrLivArea`    | 1500.72      | 1562.10      |
| `TotalBsmtSF`  | 1051.00      | 1103.23      |
| `GarageArea`   | 491.75       | 514.16       |
| `GarageYrBlt`  | 1976.89      | 1979.91      |
| `SalePrice`    | 180,642.21   | 190,370.10   |

These intervals provide a useful estimate of the **typical range** for each feature’s mean in the population.

---

### Summary

This extended classical inference step allowed us to:

- Describe the distribution and central tendency of each numerical feature
- Quantify uncertainty in the mean values via confidence intervals
- Test statistical significance of feature means
- Visualize the shape of key distributions
