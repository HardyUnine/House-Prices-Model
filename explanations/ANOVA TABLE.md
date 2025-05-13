| Term                      | F-value |  p-value | Interpretation            |
| ------------------------- | ------: | -------: | ------------------------- |
| `KitchenQual_code`        |   99.63 | 9.87e-23 |  Highly significant     |
| `GarageFinish_code`       |   68.01 | 3.62e-16 |  Highly significant     |
| `BsmtQual_code`           |   64.08 | 2.45e-15 |  Highly significant     |
| `ExterQual_code`          |   52.08 | 8.58e-13 |  Highly significant     |
| `BsmtExposure_code`       |   50.68 | 1.71e-12 |  Highly significant     |
| `HeatingQC_code`          |    0.82 |    0.364 |  Not significant         |
| `ExterQual × KitchenQual` |   16.06 | 6.46e-05 |  Significant interaction |
| `BsmtQual × BsmtExposure` |   17.28 | 3.42e-05 |  Significant interaction |
| `KitchenQual × BsmtQual`  |    5.15 |   0.0233 |  Mild interaction        |
| `ExterQual × HeatingQC`   |    4.40 |   0.0360 |  Mild interaction        |
| *Other interactions*      |  < 1.00 |   > 0.30 |  Not significant         |
