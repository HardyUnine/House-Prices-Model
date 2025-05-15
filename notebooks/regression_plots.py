import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

df = pd.read_csv("data/clean_reclean/numerical_cleaned.csv")
df = df.drop(columns=[col for col in df.columns if "Id" in col or col == "SalePrice.1"])
df_clean = df.dropna()

# Prepare features and targets
X_full = df_clean.drop(columns=["SalePrice"])
y_orig = df_clean["SalePrice"]
y_log = np.log(y_orig)

# Drop TotRmsAbvGrd for models 2 and 3 -> bad predictor
# (we'll keep it for model 1)
X_reduced = df_clean.drop(columns=["SalePrice", "TotRmsAbvGrd"])

# Add constants
X_full_const = sm.add_constant(X_full)
X_reduced_const = sm.add_constant(X_reduced)

# === Model 1: Full Numeric ===
model_full = sm.OLS(y_orig, X_full_const).fit()

# === Model 2: Cleaned Numeric ===
model_cleaned_raw = sm.OLS(y_orig, X_reduced_const).fit()
resid_cleaned = model_cleaned_raw.resid
mask_no_outliers = np.abs(resid_cleaned) <= 3 * resid_cleaned.std()
X_cleaned = X_reduced_const[mask_no_outliers]
y_cleaned = y_orig[mask_no_outliers]
model_cleaned = sm.OLS(y_cleaned, X_cleaned).fit()

# === Model 3: Log-Transformed Final ===
y_log_cleaned = y_log[mask_no_outliers]
model_log = sm.OLS(y_log_cleaned, X_cleaned).fit()


# === Plotting function for 4 diagnostic plots ===
def full_diagnostics(model, y_true, y_pred, title_prefix):
    residuals = y_true - y_pred

    # 1. Actual vs Predicted
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_pred, y=y_true, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f"{title_prefix}: Actual vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Residuals vs Fitted
    plt.figure(figsize=(6, 5))
    sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.title(f"{title_prefix}: Residuals vs Fitted")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Histogram of Residuals
    plt.figure(figsize=(6, 5))
    sns.histplot(residuals, kde=True)
    plt.title(f"{title_prefix}: Residual Histogram")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4. Manual QQ Plot with Fitted Line
    sorted_resid = np.sort(residuals)
    theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_resid)))
    fit = np.polyfit(theoretical_q, sorted_resid, 1)
    line = np.poly1d(fit)(theoretical_q)

    plt.figure(figsize=(6, 5))
    plt.scatter(theoretical_q, sorted_resid, alpha=0.6)
    plt.plot(theoretical_q, line, 'r--', label=f'Fit Line: y={fit[0]:.2f}x + {fit[1]:.2f}')
    plt.title(f"{title_prefix}: QQ Plot (Manual)")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === Generate plots for each model ===
full_diagnostics(model_full, y_orig, model_full.fittedvalues, "Model 1 - Full Numeric")
full_diagnostics(model_cleaned, y_cleaned, model_cleaned.fittedvalues, "Model 2 - Cleaned Numeric")
full_diagnostics(model_log, y_log_cleaned, model_log.fittedvalues, "Model 3 - Log-Transformed")
