import pandas as pd
import matplotlib.pyplot as plt

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

