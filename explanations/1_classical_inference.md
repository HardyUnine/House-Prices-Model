# Classical Statistical Inference

In this section, we perform classical statistical inference on the numerical features of the housing dataset, with a focus on the `SalePrice` variable.

### Descriptive Statistics
We start by computing basic summary statistics (mean, standard deviation, min, max, quartiles) to understand the distribution of numerical variables, especially the sale price.

### Visualization
We visualize the distribution of `SalePrice` using a histogram and a Q-Q plot to check for normality. This helps us understand whether it's appropriate to use parametric methods like t-tests and confidence intervals.

### Confidence Interval
We calculate a 95% confidence interval for the mean of `SalePrice` using the t-distribution. This interval provides a range of plausible values for the population mean based on our sample data.

### Hypothesis Testing
We test the null hypothesis that the average sale price is $180,000 against the alternative that it's greater than $180,000. A one-tailed t-test is used for this purpose, and the resulting p-value tells us whether the difference is statistically significant.

### Confidence Intervals for All Numerical Features
Finally, we compute 95% confidence intervals for the mean of all numerical features in the dataset to understand the typical range of values for each variable.

This analysis provides foundational insights into the housing data and sets the stage for more complex modeling techniques like regression and ANOVA.
