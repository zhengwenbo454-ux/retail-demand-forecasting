"""Feature engineering shared by training and evaluation.

Every sales-based feature looks back at least HORIZON (28) days. When we forecast
day t+1 ... t+28 on day t, those values are already known, so nothing leaks from
the future.
"""
import pandas as pd

from config import HORIZON

LAGS = [HORIZON, HORIZON + 7, HORIZON + 14, HORIZON + 28]   # 28, 35, 42, 56 days back
WINDOWS = [7, 28, 56]                                       # rolling windows ending 28 days back
CATEGORICAL = ["item_id", "dept_id", "cat_id", "event_type_1"]

FEATURES = (
    [f"lag_{lag}" for lag in LAGS]
    + [f"roll_mean_{w}" for w in WINDOWS]
    + ["roll_std_28", "sell_price", "price_vs_8wk", "price_change_1wk",
       "dow", "dom", "month", "is_event", "snap"]
    + CATEGORICAL
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    by_item = df.groupby("id")["sales"]

    # 1) Lags: sales on the same item 28, 35, 42 and 56 days earlier.
    for lag in LAGS:
        df[f"lag_{lag}"] = by_item.shift(lag).astype("float32")

    # 2) Rolling statistics of the series shifted by 28 days.
    shifted = by_item.shift(HORIZON)
    shifted_by_item = shifted.groupby(df["id"])
    for w in WINDOWS:
        df[f"roll_mean_{w}"] = shifted_by_item.transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()).astype("float32")
    df["roll_std_28"] = shifted_by_item.transform(
        lambda s: s.rolling(28, min_periods=7).std()).astype("float32")

    # 3) Price: current price vs the item's average price over the last 8 weeks,
    #    and the change from last week (discounts show up as negative values).
    by_price = df.groupby("id")["sell_price"]
    avg_8wk = by_price.transform(lambda s: s.rolling(56, min_periods=1).mean())
    df["price_vs_8wk"] = (df["sell_price"] / avg_8wk).astype("float32")
    df["price_change_1wk"] = (df["sell_price"] / by_price.shift(7) - 1).astype("float32")

    # 4) Calendar.
    df["dow"] = df["date"].dt.dayofweek.astype("int8")   # 0 = Monday
    df["dom"] = df["date"].dt.day.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["is_event"] = df["event_name_1"].notna().astype("int8")
    df["snap"] = df["snap"].astype("int8")

    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df
