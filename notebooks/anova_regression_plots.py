import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("data/clean_reclean/ordinal_numerized_cleaned.csv")
df = df.drop(columns=[col for col in df.columns if "Id" in col or col == "SalePrice.1"])
df = df.dropna()

# Features and target
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# Add constant and fit model
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
residuals = model.resid
fitted = model.fittedvalues

# 1. Actual vs Predicted
plt.figure(figsize=(6, 5))
sns.scatterplot(x=fitted, y=y, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Ordinal Model: Actual vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Residuals vs Fittedd
plt.figure(figsize=(6, 5))
plt.scatter(fitted, residuals, alpha=0.6)
sns.regplot(x=fitted, y=residuals, scatter=False, lowess=True, line_kws={'color': 'red'})
plt.title("Ordinal Model: Residuals vs Fitted (Manual)")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Histogram of residuals
plt.figure(figsize=(6, 5))
sns.histplot(residuals, kde=True)
plt.title("Ordinal Model: Residual Histogram")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Manual QQ Plot
sorted_resid = np.sort(residuals)
theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_resid)))
fit = np.polyfit(theoretical_q, sorted_resid, 1)
line = np.poly1d(fit)(theoretical_q)

plt.figure(figsize=(6, 5))
plt.scatter(theoretical_q, sorted_resid, alpha=0.6)
plt.plot(theoretical_q, line, 'r--', label=f'Fit Line: y = {fit[0]:.2f}x + {fit[1]:.2f}')
plt.title("Ordinal Model: QQ Plot (Manual Line)")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
