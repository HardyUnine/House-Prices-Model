import pandas as pd
import statsmodels.api as sm

# Load the cleaned numerical data
df = pd.read_csv("data/clean_reclean/numerical_cleaned.csv")

# Drop ID columns and duplicated SalePrice if present
df = df.drop(columns=[col for col in df.columns if "Id" in col or col == "SalePrice.1"])

# Drop rows with missing values
df_clean = df.dropna()

# Separate features (X) and target (y)
X = df_clean.drop(columns=["SalePrice"])
y = df_clean["SalePrice"]

# Add constant to features
X_with_const = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X_with_const).fit()

# Show model summary
print(model.summary())

# Optional: Describe residuals
print("\nResiduals summary:\n", model.resid.describe())
