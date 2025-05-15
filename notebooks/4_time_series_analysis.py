import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1. Load data
df = pd.read_csv('data/clean_reclean/time_series.csv')

# 2. Create a datetime index
df['Date'] = pd.to_datetime(df['YrSold'].astype(str) + '-' + df['MoSold'].astype(str) + '-01')
df = df.sort_values('Date')

# 3. Aggregate by month (mean SalePrice per month)
monthly = df.groupby('Date')['SalePrice'].mean()

# 4. Fit SARIMA model (example order, you may tune these)
# SARIMA(p,d,q)(P,D,Q,s)
model = SARIMAX(monthly, order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit(disp=False)

# 5. Print summary and plot
print(results.summary())

plt.figure(figsize=(12,6))
plt.plot(monthly, label='Observed')
plt.plot(results.fittedvalues.dropna(), label='Fitted', color='red')
plt.legend()
plt.title('SARIMA Fit to SalePrice Time Series')
plt.show()