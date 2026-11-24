"""Step 3 - baselines, LightGBM, evaluation and figures.

Split by time (never at random):
  train : everything before the validation window
  valid : the 28 days before the test window (used for early stopping)
  test  : the last 28 days (only used once, for the final numbers)

Input : data/processed/sales_long.parquet
Output: reports/metrics.md, reports/metrics.json, reports/wape_by_dept.csv,
        reports/figures/forecast_vs_actual.png, reports/figures/feature_importance.png,
        models/lgbm.txt
Run   : python src/train.py
"""
import json

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import FIGURES, HORIZON, MODELS, PROCESSED, REPORTS, STORE  # noqa: E402
from features import FEATURES, LAGS, add_features  # noqa: E402

PARAMS = {
    "objective": "tweedie",          # many zeros + a long tail of high sales
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "metric": "rmse",
    "seed": 42,
    "verbose": -1,
}
MODELS_TO_REPORT = {
    "pred_mean28": "Last-28-day mean (baseline)",
    "pred_snaive": "Seasonal naive: same weekday last week (baseline)",
    "pred_lgbm": "LightGBM (Tweedie)",
}


def wape(y, p):
    return float(np.abs(y - p).sum() / y.sum())


def scores(y, p):
    return {"WAPE": wape(y, p),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "MAE": float(np.mean(np.abs(y - p))),
            "Bias": float((p.sum() - y.sum()) / y.sum())}


def main():
    df = add_features(pd.read_parquet(PROCESSED / "sales_long.parquet"))

    last_day = df["date"].max()
    test_start = last_day - pd.Timedelta(days=HORIZON - 1)
    valid_start = test_start - pd.Timedelta(days=HORIZON)
    print(f"valid: {valid_start:%Y-%m-%d} .. {test_start - pd.Timedelta(days=1):%Y-%m-%d}")
    print(f"test : {test_start:%Y-%m-%d} .. {last_day:%Y-%m-%d}")

    # Train only on days when the item was on sale and has a full history of lags.
    usable = df["sell_price"].notna() & df[f"lag_{max(LAGS)}"].notna()
    train = df[usable & (df["date"] < valid_start)]
    valid = df[usable & (df["date"] >= valid_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start].copy()
    print(f"rows -> train {len(train):,} | valid {len(valid):,} | test {len(test):,}")

    # 1) Fit with early stopping on the validation window.
    model = lgb.train(
        PARAMS,
        lgb.Dataset(train[FEATURES], train["sales"]),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(valid[FEATURES], valid["sales"])],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    best_rounds = model.best_iteration or 300
    print(f"best number of trees: {best_rounds}")

    # 2) Refit on train + valid with that number of trees, then forecast the test window.
    full = df[usable & (df["date"] < test_start)]
    final = lgb.train(PARAMS, lgb.Dataset(full[FEATURES], full["sales"]),
                      num_boost_round=best_rounds)
    final.save_model(str(MODELS / "lgbm.txt"))
    test["pred_lgbm"] = np.clip(final.predict(test[FEATURES]), 0, None)

    # 3) Baselines, built only from data available before the test window.
    history = df[df["date"] < test_start]
    last_28 = history[history["date"] >= test_start - pd.Timedelta(days=28)]
    test["pred_mean28"] = test["id"].map(last_28.groupby("id")["sales"].mean()).fillna(0)
    last_week = history[history["date"] >= test_start - pd.Timedelta(days=7)]
    last_week = last_week[["id", "dow", "sales"]].rename(columns={"sales": "pred_snaive"})
    test = test.merge(last_week, on=["id", "dow"], how="left")
    test["pred_snaive"] = test["pred_snaive"].fillna(0)

    # An item with no price is not on sale: every method forecasts 0 for it.
    off_sale = test["sell_price"].isna()
    for col in MODELS_TO_REPORT:
        test.loc[off_sale, col] = 0.0

    # 4) Scores.
    y = test["sales"].to_numpy(dtype=float)
    results = {name: scores(y, test[col].to_numpy(dtype=float))
               for col, name in MODELS_TO_REPORT.items()}
    lgbm = results[MODELS_TO_REPORT["pred_lgbm"]]
    best_baseline = min((n for n in results if "baseline" in n), key=lambda n: results[n]["WAPE"])
    improvement = 1 - lgbm["WAPE"] / results[best_baseline]["WAPE"]

    lines = [f"Store {STORE}, {test['id'].nunique():,} items, test window "
             f"{test_start:%Y-%m-%d} to {last_day:%Y-%m-%d} ({HORIZON} days).", "",
             "| Model | WAPE | RMSE | MAE | Bias |", "| --- | --- | --- | --- | --- |"]
    for name, s in results.items():
        lines.append(f"| {name} | {s['WAPE']:.3f} | {s['RMSE']:.3f} | {s['MAE']:.3f} | {s['Bias']:+.1%} |")
    lines += ["", f"LightGBM lowers WAPE by {improvement:.1%} compared with the best baseline ({best_baseline})."]
    (REPORTS / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORTS / "metrics.json").write_text(json.dumps(
        {"test_start": f"{test_start:%Y-%m-%d}", "test_end": f"{last_day:%Y-%m-%d}",
         "best_rounds": best_rounds, "best_baseline": best_baseline,
         "wape_improvement_vs_best_baseline": improvement,
         "results": results}, indent=2), encoding="utf-8")
    print("\n" + "\n".join(lines))

    # WAPE by department (where is the model strong or weak?).
    test["abs_err_lgbm"] = (test["sales"] - test["pred_lgbm"]).abs()
    test["abs_err_snaive"] = (test["sales"] - test["pred_snaive"]).abs()
    by_dept = test.groupby("dept_id", observed=True).agg(
        units=("sales", "sum"), err_lgbm=("abs_err_lgbm", "sum"), err_snaive=("abs_err_snaive", "sum"))
    by_dept["wape_lgbm"] = (by_dept["err_lgbm"] / by_dept["units"]).round(3)
    by_dept["wape_snaive"] = (by_dept["err_snaive"] / by_dept["units"]).round(3)
    by_dept[["units", "wape_snaive", "wape_lgbm"]].to_csv(REPORTS / "wape_by_dept.csv")
    print("\nWAPE by department:\n", by_dept[["units", "wape_snaive", "wape_lgbm"]])

    # Figure: total daily units, actual vs forecasts, with 8 weeks of history.
    context = df[(df["date"] >= test_start - pd.Timedelta(days=56)) & (df["date"] < test_start)]
    hist_total = context.groupby("date")["sales"].sum()
    totals = test.groupby("date")[["sales", "pred_snaive", "pred_lgbm"]].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hist_total.index, hist_total.values, color="#9aa3ad", lw=1, label="History")
    ax.plot(totals.index, totals["sales"], color="#1d232b", lw=1.6, label="Actual")
    ax.plot(totals.index, totals["pred_snaive"], color="#9aa3ad", lw=1.2, ls="--", label="Seasonal naive")
    ax.plot(totals.index, totals["pred_lgbm"], color="#c8262c", lw=1.6, label="LightGBM")
    ax.axvline(test_start, color="#9aa3ad", lw=0.8)
    ax.set_title(f"Store {STORE}: total units per day, 28-day forecast vs actual")
    ax.set_ylabel("Units")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "forecast_vs_actual.png", dpi=150)
    plt.close(fig)

    # Figure: which features the model relied on most (total gain).
    imp = pd.Series(final.feature_importance("gain"), index=FEATURES).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values / imp.values.sum(), color="#1f4e79")
    ax.set_xlabel("Share of total gain")
    ax.set_title("Top 15 features (LightGBM)")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved metrics to {REPORTS} and figures to {FIGURES}")


if __name__ == "__main__":
    main()
