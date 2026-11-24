"""Shared settings for every step of the project.

Change STORE or START_DATE here if you want to try another store or period.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"              # put the Kaggle CSV files here
PROCESSED = ROOT / "data" / "processed"  # created by prepare_data.py
REPORTS = ROOT / "reports"               # reports and figures that go on GitHub
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"                 # trained model (not uploaded)

STORE = "CA_1"            # one Walmart store keeps the data laptop-sized
START_DATE = "2014-01-01"  # use the last ~2.4 years of history
HORIZON = 28               # forecast 28 days ahead, as in the M5 competition

for folder in (PROCESSED, FIGURES, MODELS):
    folder.mkdir(parents=True, exist_ok=True)
