# Retail Demand Forecasting with LightGBM (Walmart M5, store CA_1)

Forecasting daily unit sales of 3,049 products 28 days ahead, after auditing the data with SQL.

**Result:** on a 28-day hold-out (25 April – 22 May 2016), LightGBM reaches a WAPE of 0.731, which is 2.2% lower than the best simple baseline (last-28-day mean) and 15.8% lower than a seasonal naive forecast.

![Forecast vs actual](reports/figures/forecast_vs_actual.png)

## Results

Store CA_1, 3,049 items, test window 2016-04-25 to 2016-05-22 (28 days).

| Model | WAPE | RMSE | MAE | Bias |
| --- | --- | --- | --- | --- |
| Last-28-day mean (baseline) | 0.748 | 2.347 | 1.174 | -4.5% |
| Seasonal naive: same weekday last week (baseline) | 0.868 | 2.737 | 1.363 | -5.6% |
| **LightGBM (Tweedie)** | 0.731 | 2.241 | 1.148 | -5.8% |

WAPE = total absolute error / total units sold (lower is better). Bias = (forecast total − actual total) / actual total.

- **Item-level daily demand is sparse.** Most items sell only a few units a day and nothing on more than half (56%) of the days they are on sale, so item-level WAPE stays around 0.7 even for the best model.
- **Every method under-forecasts.** The last-28-day mean has a bias of -4.5%, which means the test window sold more than the four weeks before it. LightGBM (-5.8%) does not pick up this short-term rise either.
- **Errors follow intermittency.** LightGBM beats the seasonal naive forecast in all seven departments. It is most accurate in FOODS_3 (WAPE 0.598), the largest department and the one with the fewest zero-sales days (44.0%), and least accurate in HOBBIES_2 (1.365) and HOUSEHOLD_2 (1.132), where 82.6% and 74.1% of on-sale days have zero sales.

Per-department results: [`reports/wape_by_dept.csv`](reports/wape_by_dept.csv).

## Data quality findings

Checks are written in SQL ([`sql/data_quality_checks.sql`](sql/data_quality_checks.sql)) and run with DuckDB.
Full report: [`reports/data_quality.md`](reports/data_quality.md).

- **No structural errors.** All 873 days are present, there are no duplicate item-days, no negative sales or non-positive prices, and every item ID matches its department and category.
- **"Not on sale" is different from "zero demand".** 75,674 item-days (2.84%) have no price, and every one of them has zero sales: the item was not on the shelf that week. 419 of the 3,049 items were launched after 1 January 2014. These rows are excluded from training and forecast as 0, so the model only learns from days when an item could actually be bought.
- **Demand is intermittent.** On days when an item is on sale, it sells nothing on 66.7% of HOBBIES, 60.9% of HOUSEHOLD and 48.6% of FOODS item-days. This is why the model uses a Tweedie objective instead of plain squared error.
- **Christmas closures.** Total store sales were 0 on 25 December 2014 and 25 December 2015. These days are kept; the event features let the model learn them.
- **Demand spikes.** 1,160 item-days across 410 items sold at least 10 units and more than 10 times the item's average. The largest is 239 units of HOUSEHOLD_1_133 on 2 October 2014, against an average of about 1 a day. Only one of the 15 largest spikes falls on an event day (Halloween); the rest are kept as-is and flagged for manual review.
- **Prices rarely change.** 1,541 items (51%) had a single price over the whole period, and only 16 items had seven or more prices. This motivated experiment B below.

![Zero sales by department](reports/figures/zero_share_by_dept.png)

## Approach

1. **Data** – M5 *sales_train_evaluation*, calendar and weekly prices for store CA_1 from 2014-01-01, reshaped to one row per item per day: 2,661,777 rows (`src/prepare_data.py`).
2. **Validation** – split by time: train → 28-day validation window (early stopping) → final 28-day test window. No random splits.
3. **Features** (`src/features.py`) – sales lags of 28–56 days and rolling means/std ending 28 days back (no information from inside the forecast window), price vs its 8-week average, weekly price change, day of week/month, events and SNAP days, item/department/category.
4. **Model** – LightGBM with a Tweedie objective (early stopping chose 176 trees), refit on train + validation before forecasting the test window (`src/train.py`).
5. **Baselines** – last-28-day mean and seasonal naive (same weekday last week).

![Feature importance](reports/figures/feature_importance.png)

The model relies mostly on recent sales levels: the 56-day and 28-day rolling means and the item itself account for over 90% of the total gain.

## Experiments

**B – Do price features help?** I removed the three price features (`sell_price`, `price_vs_8wk`, `price_change_1wk`) and retrained.

| Run | WAPE | RMSE | MAE |
| --- | --- | --- | --- |
| All features (reference) | 0.732 | 2.242 | 1.149 |
| Without price features | 0.733 | 2.240 | 1.151 |

- Removing price barely changes overall accuracy (WAPE +0.001). The price features carry only 0.8% of the model's total gain.
- The likely reason is the data itself: half of the items never change price, and for the rest the effect of past price changes is already reflected in recent sales.
- Without price, WAPE rose slightly in 5 of 7 departments, most in HOBBIES_2 (1.370 → 1.397) and HOUSEHOLD_2 (1.133 → 1.143).
- **Decision:** keep the price features. They cost little, help most departments a little, and would be needed for pricing or promotion what-if questions.

_Both runs were made on the same machine. The reference WAPE (0.732) differs from the main results table (0.731) by 0.001 because LightGBM ran on a different platform._

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Download the M5 data from Kaggle (see below) into data/raw/
python src/prepare_data.py
python src/data_quality.py
python src/train.py
```

## Project structure

```
├── data/raw/                  # Kaggle files (not in the repo)
├── reports/                   # data quality report, metrics, figures
├── sql/data_quality_checks.sql
└── src/
    ├── config.py              # store, start date, horizon
    ├── prepare_data.py        # step 1: tidy table
    ├── data_quality.py        # step 2: SQL checks + report
    ├── features.py            # feature engineering
    └── train.py               # step 3: baselines, LightGBM, evaluation
```

## Next steps

- **Intermittent-demand methods** (e.g. Croston/TSB) for HOBBIES and HOUSEHOLD, compared with LightGBM on those departments.
- **Rolling-origin backtest** over several 28-day windows, to check whether the under-forecasting seen here is consistent.
- **SHAP explanations** for individual forecasts.

## Data

[M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) (Kaggle, Walmart unit sales 2011–2016).
The data is not included in this repository: accept the competition rules on Kaggle, download the files and put
`calendar.csv`, `sell_prices.csv` and `sales_train_evaluation.csv` in `data/raw/`.
