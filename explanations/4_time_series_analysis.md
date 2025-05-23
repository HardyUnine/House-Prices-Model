# Time Series Analysis with SARIMAX

This notebook explores time series modeling techniques using SARIMAX, focusing on real estate sale price data. We analyze and forecast monthly average sale prices while incorporating exogenous variables.

## Overview
This notebook includes the following key steps:
- Time series decomposition to visualize trend, seasonality, and residuals.
- Log transformation of sale prices to stabilize variance.
- Use of SARIMAX models to capture both seasonal and non-seasonal patterns.
- Incorporation of exogenous variables (`OverallQual`, `YearBuilt`) to enhance model accuracy.
- Evaluation of forecast accuracy using relative forecast error variance.
- Visualization of forecast vs actual prices.


## Dataset
The dataset includes monthly aggregated sale prices and quality metrics from real estate transactions. We resample data to monthly frequency and compute average values before modeling.

## Model Selection
We tested multiple combinations of non-seasonal (p, d, q) and seasonal (P, D, Q, s) parameters. Model performance was evaluated using:
- AIC (Akaike Information Criterion)
- Variance of relative forecast errors
- Visual inspection of forecast vs actual price trends

## Results
Top-performing models effectively captured trend and seasonality while benefiting from the inclusion of `OverallQual` and `Yearbuilt`as exogenous regressors. Forecast plots demonstrate strong alignment with actual values in the test set.
