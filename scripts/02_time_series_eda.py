# ============================================================
# Assignment 8: Build & Evaluate 
# Time Series, Regression, and Classification Models
#
# 02_time_series_eda.py
#
# Purpose:
#   Conduct deeper exploratory time-series analysis of the
#   Store Sales data and save durable report-ready artifacts
#   for SoarDDDS8555-8-Report.Rmd.
#
# Focus:
#   - Long-term sales trend
#   - Rolling averages and volatility
#   - Weekly seasonality
#   - Annual seasonality
#   - Autocorrelation
#   - Product-family sales patterns
#   - Store-level sales patterns
#   - Promotions and sales
# ============================================================

from pathlib import Path

import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf


# ---------------------------------------------------------
# Project directories
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

STORE_DIR = DATA_DIR / "store_sales_time_series_forecasting"
HOUSE_DIR = DATA_DIR / "house_prices_advanced_regression_techniques"
CRIME_DIR = DATA_DIR / "san_francisco_crime_classification"


# ---------------------------------------------------------
# Output directories
# ---------------------------------------------------------

OUTPUT_DIR = PROJECT_DIR / "output"

FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = OUTPUT_DIR / "results"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------

ROLLING_7_DAYS = 7
ROLLING_30_DAYS = 30
ROLLING_365_DAYS = 365

TOP_FAMILIES = 10
TOP_STORES = 10


# ---------------------------------------------------------
# Load Store Sales data
# ---------------------------------------------------------

print("=" * 60)
print("02_time_series_eda.py")
print("=" * 60)

print("\nLoading Store Sales training data...")

store_train = pd.read_csv(
    STORE_DIR / "train.csv",
    parse_dates=["date"]
)

print("Training data shape:", store_train.shape)
print("Date range:")
print("Start:", store_train["date"].min())
print("End:", store_train["date"].max())


# ---------------------------------------------------------
# Daily aggregate sales
# ---------------------------------------------------------

daily_sales = (
    store_train
    .groupby("date", as_index=False)["sales"]
    .sum()
    .sort_values("date")
    .reset_index(drop=True)
)

print("\nDaily aggregate observations:", len(daily_sales))


# ---------------------------------------------------------
# Time-Series Trend and Rolling Statistics
# ---------------------------------------------------------

daily_sales["rolling_mean_7"] = (
    daily_sales["sales"]
    .rolling(ROLLING_7_DAYS)
    .mean()
)

daily_sales["rolling_mean_30"] = (
    daily_sales["sales"]
    .rolling(ROLLING_30_DAYS)
    .mean()
)

daily_sales["rolling_std_30"] = (
    daily_sales["sales"]
    .rolling(ROLLING_30_DAYS)
    .std()
)

daily_sales["rolling_mean_365"] = (
    daily_sales["sales"]
    .rolling(ROLLING_365_DAYS)
    .mean()
)

daily_sales.to_csv(
    TABLES_DIR / "store_sales_rolling_statistics.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_rolling_statistics.csv"
)


# ---------------------------------------------------------
# Trend Figure
# ---------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    daily_sales["date"],
    daily_sales["sales"],
    color="lightgray",
    linewidth=0.7,
    label="Daily sales"
)

plt.plot(
    daily_sales["date"],
    daily_sales["rolling_mean_30"],
    color="steelblue",
    linewidth=2,
    label="30-day rolling mean"
)

plt.plot(
    daily_sales["date"],
    daily_sales["rolling_mean_365"],
    color="darkorange",
    linewidth=2,
    label="365-day rolling mean"
)

plt.title("Store Sales Trend with Rolling Averages")
plt.xlabel("Date")
plt.ylabel("Total Daily Sales")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "store_sales_rolling_trends.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR / "store_sales_rolling_trends.png"
)


# ---------------------------------------------------------
# Month and Year Seasonality
# ---------------------------------------------------------

daily_sales["year"] = daily_sales["date"].dt.year
daily_sales["month"] = daily_sales["date"].dt.month
daily_sales["month_name"] = daily_sales["date"].dt.month_name()
daily_sales["day_of_week"] = daily_sales["date"].dt.day_name()
daily_sales["day_of_week_num"] = daily_sales["date"].dt.dayofweek


# ---------------------------------------------------------
# Monthly Sales Summary
# ---------------------------------------------------------

monthly_sales = (
    daily_sales
    .groupby(
        ["year", "month"],
        as_index=False
    )["sales"]
    .sum()
)

monthly_sales.to_csv(
    TABLES_DIR / "store_sales_monthly_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_monthly_summary.csv"
)


# ---------------------------------------------------------
# Monthly Seasonality
# ---------------------------------------------------------

monthly_seasonality = (
    daily_sales
    .groupby("month")["sales"]
    .agg(["mean", "median", "std"])
    .reset_index()
)

monthly_seasonality.to_csv(
    TABLES_DIR / "store_sales_monthly_seasonality.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_monthly_seasonality.csv"
)


plt.figure(figsize=(10, 5))

sns.barplot(
    data=monthly_seasonality,
    x="month",
    y="mean",
    color="steelblue"
)

plt.title("Average Daily Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Daily Sales")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "store_sales_monthly_seasonality.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR / "store_sales_monthly_seasonality.png"
)


# ---------------------------------------------------------
# Year-over-Year Monthly Pattern
# ---------------------------------------------------------

monthly_pivot = monthly_sales.pivot(
    index="month",
    columns="year",
    values="sales"
)

monthly_pivot.to_csv(
    TABLES_DIR / "store_sales_monthly_year_comparison.csv",
    index=True
)


print(
    "Saved table:",
    TABLES_DIR / "store_sales_monthly_year_comparison.csv"
)


plt.figure(figsize=(12, 6))

for year in monthly_pivot.columns:
    plt.plot(
        monthly_pivot.index,
        monthly_pivot[year],
        marker="o",
        linewidth=1.5,
        label=str(year)
    )

plt.title("Monthly Store Sales by Year")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(range(1, 13))
plt.legend(title="Year")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "store_sales_monthly_year_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR / "store_sales_monthly_year_comparison.png"
)


# ---------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------

acf_values = pd.DataFrame({
    "lag": range(1, 31),
    "autocorrelation": [
        daily_sales["sales"].autocorr(lag=lag)
        for lag in range(1, 31)
    ]
})

acf_values.to_csv(
    TABLES_DIR / "store_sales_autocorrelation.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_autocorrelation.csv"
)


plt.figure(figsize=(12, 6))

plot_acf(
    daily_sales["sales"],
    lags=30,
    ax=plt.gca(),
    zero=False
)

plt.title("Autocorrelation of Daily Store Sales")
plt.xlabel("Lag (days)")
plt.ylabel("Autocorrelation")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "store_sales_autocorrelation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR / "store_sales_autocorrelation.png"
)


# ---------------------------------------------------------
# Product Family Sales
# ---------------------------------------------------------

family_sales = (
    store_train
    .groupby("family", as_index=False)["sales"]
    .sum()
    .sort_values("sales", ascending=False)
)

family_sales["sales_share"] = (
    family_sales["sales"]
    / family_sales["sales"].sum()
)

family_sales.to_csv(
    TABLES_DIR / "store_sales_family_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_family_summary.csv"
)


top_families = family_sales.head(TOP_FAMILIES)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=top_families,
    x="sales",
    y="family",
    color="steelblue"
)

plt.title("Top Product Families by Total Sales")
plt.xlabel("Total Sales")
plt.ylabel("Product Family")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "store_sales_top_product_families.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR / "store_sales_top_product_families.png"
)


# ---------------------------------------------------------
# Store Sales
# ---------------------------------------------------------

store_sales = (
    store_train
    .groupby("store_nbr", as_index=False)["sales"]
    .sum()
    .sort_values("sales", ascending=False)
)

store_sales["sales_share"] = (
    store_sales["sales"]
    / store_sales["sales"].sum()
)

store_sales.to_csv(
    TABLES_DIR / "store_sales_store_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_store_summary.csv"
)


top_stores = store_sales.head(TOP_STORES)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_stores,
    x="sales",
    y="store_nbr",
    color="darkorange"
)

plt.title("Top Stores by Total Sales")
plt.xlabel("Total Sales")
plt.ylabel("Store Number")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "store_sales_top_stores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR / "store_sales_top_stores.png"
)


# ---------------------------------------------------------
# Promotion and Sales Relationship
# ---------------------------------------------------------

promotion_counts = (
    store_train
    .groupby("date")
    .size()
    .rename("daily_observations")
    .reset_index()
)

promotion_daily = (
    store_train
    .groupby("date", as_index=False)
    .agg(
        sales=("sales", "sum"),
        total_onpromotion=("onpromotion", "sum"),
        mean_onpromotion=("onpromotion", "mean")
    )
    .merge(
        promotion_counts,
        on="date",
        how="left"
    )
)

promotion_daily["mean_items_onpromotion"] = np.where(
    promotion_daily["daily_observations"] > 0,
    promotion_daily["total_onpromotion"]
    / promotion_daily["daily_observations"],
    np.nan
)


promotion_correlation = (
    promotion_daily[
        [
            "sales",
            "total_onpromotion",
            "mean_items_onpromotion"
        ]
    ]
    .corr()
)

promotion_daily.to_csv(
    TABLES_DIR / "store_sales_promotion_daily.csv",
    index=False
)

promotion_correlation.to_csv(
    TABLES_DIR / "store_sales_promotion_correlation.csv"
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_promotion_daily.csv"
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_promotion_correlation.csv"
)


plt.figure(figsize=(10, 6))

plt.scatter(
    promotion_daily["total_onpromotion"],
    promotion_daily["sales"],
    alpha=0.35,
    s=15
)

plt.title("Daily Sales and Promotion Activity")
plt.xlabel("Total Items on Promotion")
plt.ylabel("Total Daily Sales")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "store_sales_promotion_relationship.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR / "store_sales_promotion_relationship.png"
)


# ---------------------------------------------------------
# EDA Results Summary
# ---------------------------------------------------------

# ---------------------------------------------------------
# EDA Results Summary
# ---------------------------------------------------------

eda_results = {
    "daily_observations": int(len(daily_sales)),
    "date_start": str(daily_sales["date"].min().date()),
    "date_end": str(daily_sales["date"].max().date()),
    "mean_daily_sales": float(daily_sales["sales"].mean()),
    "median_daily_sales": float(daily_sales["sales"].median()),
    "std_daily_sales": float(daily_sales["sales"].std()),
    "min_daily_sales": float(daily_sales["sales"].min()),
    "max_daily_sales": float(daily_sales["sales"].max()),

    "top_product_family": str(
        family_sales.iloc[0]["family"]
    ),

    "top_product_family_sales_share": float(
        family_sales.iloc[0]["sales_share"]
    ),

    "top_store": int(
        store_sales.iloc[0]["store_nbr"]
    ),

    "top_store_sales_share": float(
        store_sales.iloc[0]["sales_share"]
    ),

    "max_autocorrelation_lag_1_to_30": int(
        acf_values.loc[
            acf_values["autocorrelation"].idxmax(),
            "lag"
        ]
    ),

    "max_autocorrelation_value": float(
        acf_values["autocorrelation"].max()
    ),

    "sales_promotion_correlation": float(
        promotion_correlation.loc[
            "sales",
            "total_onpromotion"
        ]
    )
}


with open(
    RESULTS_DIR / "store_sales_eda_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        eda_results,
        file,
        indent=4
    )

print(
    "Saved results:",
    RESULTS_DIR / "store_sales_eda_results.json"
)


# ---------------------------------------------------------
# Completion
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("02_time_series_eda.py completed successfully")
print("=" * 60)
