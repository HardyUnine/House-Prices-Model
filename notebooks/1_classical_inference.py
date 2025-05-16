# 1_classical_inference.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Load the numerical data
numerical_df = pd.read_csv('data/clean_reclean/numerical_cleaned.csv')  # adjust path if needed
numerical_df = numerical_df.dropna()

# --- Descriptive statistics ---
desc_stats = numerical_df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
print("Descriptive Statistics:\n", desc_stats)

# --- Confidence intervals for all numerical features ---
def feature_confidence_intervals(df, confidence=0.95):
    results = {}
    for col in df.columns:
        x = df[col]
        n = len(x)
        mean = x.mean()
        std = x.std()
        ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=std/np.sqrt(n))
        results[col] = (round(ci[0], 2), round(ci[1], 2))
    return pd.DataFrame(results, index=[f'{int(confidence*100)}% CI Lower', f'{int(confidence*100)}% CI Upper']).T

ci_table = feature_confidence_intervals(numerical_df)
print("\n95% Confidence Intervals for Numerical Features:")
print(ci_table)

# --- Hypothesis Tests ---
# Test if mean of each feature is significantly different from a reference (e.g., population mean = 0)
print("\nHypothesis Tests (H0: mean = 0):")
for col in numerical_df.columns:
    t_stat, p_value = stats.ttest_1samp(numerical_df[col], popmean=0)
    print(f"{col}: t = {t_stat:.2f}, p = {p_value:.4f}")

# --- Visuals for SalePrice ---
plt.figure(figsize=(8, 5))
sns.histplot(numerical_df['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.grid(True)
plt.tight_layout()
plt.show()

sm.qqplot(numerical_df['SalePrice'], line='s')
plt.title("QQ Plot of SalePrice")
plt.show()

# --- Optional: QQ plots for other features ---
# You can limit this to just a few key variables to avoid too many plots
for col in ['GrLivArea', 'TotalBsmtSF', 'LotArea']:
    sm.qqplot(numerical_df[col], line='s')
    plt.title(f"QQ Plot of {col}")
    plt.show()
