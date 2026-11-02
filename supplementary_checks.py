"""Supplementary validation checks for the APMC capstone (to be committed beside main.py).

Covers four things the main script does not: duplicate audit, ADF stationarity test,
within-commodity correlation, and a seasonal-naive benchmark for the onion forecast.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

CSV = "APMC_Macroeconomic_Environmental_Price_Volatility_Dataset.csv"
PRICE, COST = "Min_Wholesale_Price_Rs_Qtl", "Cost_of_Production_Rs_Qtl"


def load(path=CSV):
    return pd.read_csv(path, parse_dates=["Date"])


def duplicate_audit(df):
    full = int(df.duplicated().sum())
    keyed = int(df.duplicated(["Date", "APMC_Mandi", "Commodity"]).sum())
    ordered = bool(
        ((df["Min_Wholesale_Price_Rs_Qtl"] <= df["Modal_Wholesale_Price_Rs_Qtl"])
         & (df["Modal_Wholesale_Price_Rs_Qtl"] <= df["Max_Wholesale_Price_Rs_Qtl"])).all()
    )
    sizes = df.groupby(["APMC_Mandi", "Commodity"]).size()
    return {
        "full_row_duplicates": full,
        "key_duplicates": keyed,
        "min<=modal<=max_everywhere": ordered,
        "series": int(len(sizes)),
        "months_per_series": sorted(set(int(v) for v in sizes)),
    }


def onion_series(df):
    sub = df[(df["APMC_Mandi"] == "Lasalgaon APMC") & (df["Commodity"] == "Onion")]
    return sub.set_index("Date").asfreq("MS")


def adf_report(series):
    out = {}
    for label, s in (("level", series), ("first_difference", series.diff().dropna())):
        stat, p, *_ = adfuller(s)
        out[label] = {"adf_stat": round(stat, 3), "p_value": round(p, 6)}
    return out


def within_commodity_corr(df):
    per = df.groupby("Commodity").apply(lambda g: g[PRICE].corr(g[COST]))
    return per.describe().round(2).to_dict()


def seasonal_naive_benchmark(series, split="2025-01-01"):
    train, test = series[series.index < split], series[series.index >= split]
    pred = train.iloc[-12:].to_numpy()
    err = np.abs(test.to_numpy() - pred)
    return {"MAE": round(err.mean(), 2), "MAPE_pct": round((err / test.to_numpy()).mean() * 100, 2)}


if __name__ == "__main__":
    data = load()
    onion = onion_series(data)
    print("Duplicates:", duplicate_audit(data))
    print("ADF:", adf_report(onion[PRICE]))
    print("Within-commodity r(price, cost):", within_commodity_corr(data))
    print("Seasonal-naive benchmark:", seasonal_naive_benchmark(onion[PRICE]))
