# 1_classical_inference.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the numerical data
numerical_df = pd.read_csv('data/clean/numerical.csv')  # adjust path if needed

# Drop any rows with missing values (you can also impute instead)
numerical_df = numerical_df.dropna()

# Basic descriptive statistics
desc_stats = numerical_df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
print("Descriptive Statistics:\n", desc_stats)

# Histogram of SalePrice
plt.figure(figsize=(8, 5))
sns.histplot(numerical_df['SalePrice'], kde=True)
plt.title('Distribution of Sale Price')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# QQ-Plot to assess normality
import statsmodels.api as sm
sm.qqplot(numerical_df['SalePrice'], line='s')
plt.title("QQ Plot of SalePrice")
plt.show()

# 95% Confidence Interval for SalePrice mean
sample_mean = numerical_df['SalePrice'].mean()
sample_std = numerical_df['SalePrice'].std()
n = len(numerical_df['SalePrice'])
conf_int = stats.t.interval(0.95, df=n-1, loc=sample_mean, scale=sample_std/np.sqrt(n))

print(f"\n95% Confidence Interval for SalePrice Mean: {conf_int[0]:,.2f} to {conf_int[1]:,.2f}")

# Hypothesis Test: Is the average SalePrice > $180,000?
# H0: mu = 180000, H1: mu > 180000
t_stat, p_value = stats.ttest_1samp(numerical_df['SalePrice'], popmean=180000)

print(f"\nT-statistic: {t_stat:.3f}")
print(f"One-tailed p-value (mean > 180000): {p_value / 2:.4f}")

if t_stat > 0 and (p_value / 2) < 0.05:
    print("Result: Reject H0 — The average SalePrice is significantly greater than $180,000.")
else:
    print("Result: Fail to reject H0 — No significant evidence that SalePrice > $180,000.")

# Optional: Compute confidence intervals for other features
def feature_confidence_intervals(df, confidence=0.95):
    results = {}
    for col in df.columns:
        data = df[col]
        mean = data.mean()
        std = data.std()
        n = len(data)
        ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=std/np.sqrt(n))
        results[col] = (round(ci[0], 2), round(ci[1], 2))
    return pd.DataFrame(results, index=[f'{int(confidence*100)}% CI Lower', f'{int(confidence*100)}% CI Upper']).T

ci_table = feature_confidence_intervals(numerical_df)
print("\nConfidence Intervals for Numerical Features:")
print(ci_table)
