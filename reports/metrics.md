Store CA_1, 3,049 items, test window 2016-04-25 to 2016-05-22 (28 days).

| Model | WAPE | RMSE | MAE | Bias |
| --- | --- | --- | --- | --- |
| Last-28-day mean (baseline) | 0.748 | 2.347 | 1.174 | -4.5% |
| Seasonal naive: same weekday last week (baseline) | 0.868 | 2.737 | 1.363 | -5.6% |
| LightGBM (Tweedie) | 0.731 | 2.241 | 1.148 | -5.8% |

LightGBM lowers WAPE by 2.2% compared with the best baseline (Last-28-day mean (baseline)).