import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# Load your CSV file
df = pd.read_csv('data/clean_reclean/time_series.csv')  # Adjust the path if needed

# Create a proper datetime column
df['Date'] = pd.to_datetime(df['YrSold'].astype(str) + '-' + df['MoSold'].astype(str) + '-01')

# Aggregate average SalePrice per month
monthly_avg = df.groupby('Date')['SalePrice'].mean().reset_index()
monthly_avg.columns = ['Date', 'AverageSalePrice']  # Rename for consistency with AverageSalePrice example

# Set datetime as index
monthly_avg.set_index('Date', inplace=True)
monthly_avg.index.name = 'Date'

# Plot
plt.figure(figsize=(10, 4))
plt.plot(monthly_avg, label='Average Sale Price', marker='o', linestyle='-', color='blue')
plt.grid(True, alpha=0.3)
plt.title('Monthly Average SalePrice')
plt.xlabel('Date')
plt.ylabel('SalePrice')
plt.legend()
plt.show()

monthly_diff = monthly_avg['AverageSalePrice'].diff().dropna()

plt.figure(figsize=(10, 4))
plt.plot(monthly_diff.index, monthly_diff.values, label='First Difference', marker='o', linestyle='-', color='blue')
plt.grid(True, alpha=0.3)
plt.title('First Difference of Average Sale Price')
plt.xlabel('Date')
plt.ylabel('Differenced Sale Price')
plt.legend()
plt.show()

X_orig = np.arange(len(df)).reshape(-1,1)
X_orig_const = sm.add_constant(X_orig)
model_orig = sm.OLS(df.values, X_orig_const).fit()
original_p_values = model_orig.pvalues[1]
print(f"Original series trend p-value: {original_p_values:.4f}")

X_diff = np.arange(len(monthly_diff)).reshape(-1,1)
X_diff_const = sm.add_constant(X_diff)
model_diff = sm.OLS(monthly_diff.values, X_diff_const).fit()
differenced_p_values = model_diff.pvalues[1]
print(f"First differenced series trend p-value: {differenced_p_values:.4f}")