# Ceci est le fichier du rapport
je pense que le mieux c'est de se baser sur les explications de chaque numéro pour ensuite le compléter 
## I Problem statement 

## II Data clearing (:clownface:)

## III Analysis trough statistical inference

## IV Analysis trough ANOVA/DOE

## V Analysis trough time series (very efficient)
At first we want to see what type of variables is each variables and how much of them have a value (Eg. only 91 houses have a non-null value for Alley). This will serve us later when using exogenous variables.

Afterwards we want to see the `Monthly Average SalePrice` evolution on time. We can not observe anything very significant yet. We then calculate the difference between each `Monthly Average SalePrice`, which show us that there is a lot of fluctations and there doesn't seem to have anything constant.

Using an import from `Statsmodels` we get to see the Trend, Seasonal and Residuals. The Trend is going downward, from the start where the `Monthly Average SalePrice` was at 185'000$ to the end where it is a bit below 175'00. The Seasonal shows us a clear pattern repeating every year (which will induce our choice of 12 for the time series). It is almost too cleanly repeating...

Getting the `P-value` will


## VI Our inexistent novel method