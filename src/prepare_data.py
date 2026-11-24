"""Step 1 - turn the raw M5 files into one tidy table for a single store.

Input : data/raw/sales_train_evaluation.csv, calendar.csv, sell_prices.csv
Output: data/processed/sales_long.parquet  (one row per item per day)
Run   : python src/prepare_data.py
"""
import pandas as pd

from config import PROCESSED, RAW, START_DATE, STORE

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def main():
    calendar = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    keep_days = set(calendar.loc[calendar["date"] >= START_DATE, "d"])

    # The sales file is "wide": one column per day (d_1 ... d_1941).
    # Read only the days we need, stored as small integers to save memory.
    header = pd.read_csv(RAW / "sales_train_evaluation.csv", nrows=0).columns
    day_cols = [c for c in header if c in keep_days]
    sales = pd.read_csv(
        RAW / "sales_train_evaluation.csv",
        usecols=ID_COLS + day_cols,
        dtype={c: "int16" for c in day_cols},
    )
    sales = sales[sales["store_id"] == STORE]
    print(f"Store {STORE}: {len(sales):,} items x {len(day_cols):,} days")

    # Wide -> long: one row per item per day.
    long = sales.melt(id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="sales")

    state = STORE.split("_")[0]  # "CA" for CA_1
    cal_cols = ["d", "date", "wm_yr_wk", "wday", "month", "year",
                "event_name_1", "event_type_1", f"snap_{state}"]
    long = long.merge(calendar[cal_cols], on="d", how="left")
    long = long.rename(columns={f"snap_{state}": "snap"})

    # Prices are weekly; a missing price means the item was not on sale that week.
    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[prices["store_id"] == STORE][["item_id", "wm_yr_wk", "sell_price"]]
    long = long.merge(prices, on=["item_id", "wm_yr_wk"], how="left")

    long = long.sort_values(["id", "date"]).reset_index(drop=True)
    out = PROCESSED / "sales_long.parquet"
    long.to_parquet(out, index=False)
    print(f"Saved {len(long):,} rows from {long['date'].min():%Y-%m-%d} "
          f"to {long['date'].max():%Y-%m-%d} -> {out}")


if __name__ == "__main__":
    main()
