# ============================================================
# Assignment 8: Build & Evaluate 
# Time Series, Regression, and Classification Models
#
# 01_data_inspection.py
#
# Purpose:
#   Inspect the Kaggle training/test data and save durable
#   report-ready artifacts for Report.Rmd.
#
#   This script performs DATA INSPECTION ONLY.
#   Modeling is handled in subsequent scripts.
# ============================================================

from pathlib import Path
import json

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Project directories
# ---------------------------------------------------------

# __file__ = ...\SoarDDDS8555-8\scripts\01_data_inspection.py
# .parent = ...\SoarDDDS8555-8\scripts
# .parent.parent = ...\SoarDDDS8555-8

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
# General settings
# ---------------------------------------------------------

FIGURE_DPI = 300


# ---------------------------------------------------------
# Helper function: save figures
# ---------------------------------------------------------

def save_figure(filename):
    """
    Save the current matplotlib figure as a high-resolution PNG.
    """
    path = FIGURES_DIR / filename

    plt.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved figure:", path)


# ---------------------------------------------------------
# Project/Data Directory Inspection
# ---------------------------------------------------------

print("Project directory:", PROJECT_DIR)
print("Data directory:", DATA_DIR)
print("Store directory:", STORE_DIR)

print("\nDirectory checks:")
print("Project exists:", PROJECT_DIR.exists())
print("Data exists:", DATA_DIR.exists())
print("Store Sales exists:", STORE_DIR.exists())
print("House Prices exists:", HOUSE_DIR.exists())
print("Crime exists:", CRIME_DIR.exists())


print("\nStore Sales files:")

for file in sorted(STORE_DIR.iterdir()):
    print(file.name)


# ---------------------------------------------------------
# Store Sales: Initial Data Inspection
# ---------------------------------------------------------

store_train_sample = pd.read_csv(
    STORE_DIR / "train.csv",
    nrows=10
)

print("\nStore Sales training sample:")
print(store_train_sample)

print("\nColumn names:")
print(store_train_sample.columns.tolist())

print("\nData types:")
print(store_train_sample.dtypes)


# ---------------------------------------------------------
# Store Sales: Load Training Data
# ---------------------------------------------------------

store_train = pd.read_csv(
    STORE_DIR / "train.csv",
    parse_dates=["date"]
)

print("\nTraining data shape:")
print(store_train.shape)

print("\nTraining data information:")
print(store_train.info())

print("\nDate range:")
print("Start:", store_train["date"].min())
print("End:", store_train["date"].max())

print("\nNumber of stores:")
print(store_train["store_nbr"].nunique())

print("\nNumber of product families:")
print(store_train["family"].nunique())

print("\nProduct families:")
print(sorted(store_train["family"].unique()))

print("\nMissing values:")
print(store_train.isna().sum())


# ---------------------------------------------------------
# Save: Dataset Summary
# ---------------------------------------------------------

dataset_summary = pd.DataFrame({
    "metric": [
        "training_rows",
        "training_columns",
        "training_start",
        "training_end",
        "number_of_stores",
        "number_of_product_families",
        "unique_store_family_combinations"
    ],
    "value": [
        store_train.shape[0],
        store_train.shape[1],
        str(store_train["date"].min().date()),
        str(store_train["date"].max().date()),
        store_train["store_nbr"].nunique(),
        store_train["family"].nunique(),
        len(
            store_train[["store_nbr", "family"]]
            .drop_duplicates()
        )
    ]
})

dataset_summary.to_csv(
    TABLES_DIR / "store_sales_dataset_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_dataset_summary.csv"
)


# ---------------------------------------------------------
# Save: Missing Values
# ---------------------------------------------------------

missing_values = (
    store_train
    .isna()
    .sum()
    .reset_index()
)

missing_values.columns = [
    "variable",
    "missing_count"
]

missing_values["missing_percent"] = (
    missing_values["missing_count"]
    / len(store_train)
    * 100
)

missing_values.to_csv(
    TABLES_DIR / "store_sales_missing_values.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_missing_values.csv"
)


# ---------------------------------------------------------
# Store Sales: Time-Series Structure
# ---------------------------------------------------------

daily_sales = (
    store_train
    .groupby("date", as_index=False)["sales"]
    .sum()
    .sort_values("date")
)

print("\nDaily aggregated sales:")
print(daily_sales.head())

print("\nDaily sales shape:")
print(daily_sales.shape)

print("\nDaily sales summary:")
print(daily_sales["sales"].describe())


# ---------------------------------------------------------
# Save: Daily Sales Data
# ---------------------------------------------------------

daily_sales.to_csv(
    TABLES_DIR / "store_sales_daily_aggregate.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_daily_aggregate.csv"
)


# ---------------------------------------------------------
# Save: Daily Sales Summary
# ---------------------------------------------------------

daily_sales_summary = (
    daily_sales["sales"]
    .describe()
    .reset_index()
)

daily_sales_summary.columns = [
    "statistic",
    "sales"
]

daily_sales_summary.to_csv(
    TABLES_DIR / "store_sales_daily_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_daily_summary.csv"
)


# ---------------------------------------------------------
# Store Sales: Initial Time-Series Plot
# ---------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    daily_sales["date"],
    daily_sales["sales"],
    linewidth=0.8,
    color="steelblue"
)

plt.title(
    "Daily Store Sales Across All Stores and Product Families"
)

plt.xlabel("Date")
plt.ylabel("Total Sales")

plt.grid(alpha=0.3)

plt.tight_layout()

save_figure(
    "store_sales_daily_trend.png"
)


# ---------------------------------------------------------
# Store Sales: Test Set Inspection
# ---------------------------------------------------------

store_test = pd.read_csv(
    STORE_DIR / "test.csv",
    parse_dates=["date"]
)

print("\nTest data shape:")
print(store_test.shape)

print("\nTest date range:")
print("Start:", store_test["date"].min())
print("End:", store_test["date"].max())

print("\nTest columns:")
print(store_test.columns.tolist())

print("\nTest number of stores:")
print(store_test["store_nbr"].nunique())

print("\nTest number of product families:")
print(store_test["family"].nunique())

print("\nFirst five test rows:")
print(store_test.head())

print("\nLast five test rows:")
print(store_test.tail())


# ---------------------------------------------------------
# Save: Test Summary
# ---------------------------------------------------------

test_summary = pd.DataFrame({
    "metric": [
        "test_rows",
        "test_columns",
        "test_start",
        "test_end",
        "number_of_stores",
        "number_of_product_families",
        "unique_store_family_combinations"
    ],
    "value": [
        store_test.shape[0],
        store_test.shape[1],
        str(store_test["date"].min().date()),
        str(store_test["date"].max().date()),
        store_test["store_nbr"].nunique(),
        store_test["family"].nunique(),
        len(
            store_test[["store_nbr", "family"]]
            .drop_duplicates()
        )
    ]
})

test_summary.to_csv(
    TABLES_DIR / "store_sales_test_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_test_summary.csv"
)


# ---------------------------------------------------------
# Store Sales: Training/Test Date Continuity
# ---------------------------------------------------------

train_dates = pd.date_range(
    store_train["date"].min(),
    store_train["date"].max(),
    freq="D"
)

test_dates = pd.date_range(
    store_test["date"].min(),
    store_test["date"].max(),
    freq="D"
)

missing_train_dates = train_dates.difference(
    store_train["date"].unique()
)

missing_test_dates = test_dates.difference(
    store_test["date"].unique()
)

print("\nExpected training calendar days:", len(train_dates))
print(
    "Actual training calendar days:",
    store_train["date"].nunique()
)

print("\nExpected test calendar days:", len(test_dates))
print(
    "Actual test calendar days:",
    store_test["date"].nunique()
)

print("\nTraining dates missing from calendar:")
print(missing_train_dates)

print("\nTest dates missing from calendar:")
print(missing_test_dates)


# ---------------------------------------------------------
# Save: Calendar Continuity
# ---------------------------------------------------------

calendar_summary = pd.DataFrame({
    "dataset": [
        "training",
        "test"
    ],
    "expected_calendar_days": [
        len(train_dates),
        len(test_dates)
    ],
    "observed_calendar_days": [
        store_train["date"].nunique(),
        store_test["date"].nunique()
    ],
    "missing_calendar_days": [
        len(missing_train_dates),
        len(missing_test_dates)
    ]
})

calendar_summary.to_csv(
    TABLES_DIR / "store_sales_calendar_summary.csv",
    index=False
)

missing_calendar_dates = pd.DataFrame({
    "dataset": (
        ["training"] * len(missing_train_dates)
        + ["test"] * len(missing_test_dates)
    ),
    "date": (
        list(missing_train_dates)
        + list(missing_test_dates)
    )
})

missing_calendar_dates.to_csv(
    TABLES_DIR / "store_sales_missing_calendar_dates.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_calendar_summary.csv"
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_missing_calendar_dates.csv"
)


# ---------------------------------------------------------
# Store Sales: Store/Family Combinations
# ---------------------------------------------------------

train_combinations = (
    store_train[["store_nbr", "family"]]
    .drop_duplicates()
)

test_combinations = (
    store_test[["store_nbr", "family"]]
    .drop_duplicates()
)

combinations_match = (
    set(map(tuple, train_combinations.values))
    ==
    set(map(tuple, test_combinations.values))
)

print(
    "\nTraining store-family combinations:",
    len(train_combinations)
)

print(
    "Test store-family combinations:",
    len(test_combinations)
)

print(
    "\nTraining/test combinations identical:",
    combinations_match
)


# ---------------------------------------------------------
# Save: Store/Family Combination Summary
# ---------------------------------------------------------

combination_summary = pd.DataFrame({
    "metric": [
        "training_combinations",
        "test_combinations",
        "combinations_identical"
    ],
    "value": [
        len(train_combinations),
        len(test_combinations),
        combinations_match
    ]
})

combination_summary.to_csv(
    TABLES_DIR / "store_sales_store_family_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_store_family_summary.csv"
)


# ---------------------------------------------------------
# Store Sales: Holiday Investigation
# ---------------------------------------------------------

holidays = pd.read_csv(
    STORE_DIR / "holidays_events.csv",
    parse_dates=["date"]
)

print("\nHoliday/event data shape:")
print(holidays.shape)

print("\nHoliday/event columns:")
print(holidays.columns.tolist())


# ---------------------------------------------------------
# Christmas Investigation
# ---------------------------------------------------------

christmas = (
    holidays[
        (holidays["date"].dt.month == 12)
        &
        (holidays["date"].dt.day == 25)
    ]
    .sort_values("date")
)

print("\nChristmas observations:")
print(christmas)


# ---------------------------------------------------------
# Save: Christmas Observations
# ---------------------------------------------------------

christmas.to_csv(
    TABLES_DIR / "store_sales_christmas_holidays.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_christmas_holidays.csv"
)


# ---------------------------------------------------------
# New Year's Day Investigation
# ---------------------------------------------------------

new_years = daily_sales[
    (daily_sales["date"].dt.month == 1)
    &
    (daily_sales["date"].dt.day == 1)
].copy()

print("\nNew Year's Day aggregate sales:")
print(new_years)


# January 2 comparison

jan2 = daily_sales[
    (daily_sales["date"].dt.month == 1)
    &
    (daily_sales["date"].dt.day == 2)
].copy()

print("\nJanuary 1 sales:")
print(new_years)

print("\nJanuary 2 sales:")
print(jan2)


# ---------------------------------------------------------
# Save: New Year's Day Comparison
# ---------------------------------------------------------

# ---------------------------------------------------------
# Store Sales: New Year's Day Comparison
# ---------------------------------------------------------

new_year_comparison = (
    new_years[["date", "sales"]]
    .rename(
        columns={
            "date": "january_1_date",
            "sales": "january_1_sales"
        }
    )
)

jan2_comparison = (
    jan2[["date", "sales"]]
    .assign(
        january_1_date=lambda x:
            x["date"] - pd.Timedelta(days=1)
    )
    .rename(
        columns={
            "date": "january_2_date",
            "sales": "january_2_sales"
        }
    )
)

new_year_comparison = (
    new_year_comparison
    .merge(
        jan2_comparison[
            [
                "january_1_date",
                "january_2_date",
                "january_2_sales"
            ]
        ],
        on="january_1_date",
        how="left"
    )
)

# ---------------------------------------------------------
# Store Sales: New Year's Day Comparison Metrics
# ---------------------------------------------------------

new_year_comparison["sales_difference"] = (
    new_year_comparison["january_2_sales"]
    - new_year_comparison["january_1_sales"]
)

# Protect against division by zero.
new_year_comparison["sales_ratio_jan2_to_jan1"] = np.where(
    new_year_comparison["january_1_sales"] != 0,
    (
        new_year_comparison["january_2_sales"]
        / new_year_comparison["january_1_sales"]
    ),
    np.nan
)

# Protect against division by zero.
new_year_comparison["sales_percent_change"] = np.where(
    new_year_comparison["january_1_sales"] != 0,
    (
        (
            new_year_comparison["january_2_sales"]
            - new_year_comparison["january_1_sales"]
        )
        / new_year_comparison["january_1_sales"]
        * 100
    ),
    np.nan
)

print("\nNew Year's Day vs. January 2:")
print(new_year_comparison)

new_year_comparison.to_csv(
    TABLES_DIR / "store_sales_new_year_comparison.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_new_year_comparison.csv"
)



# ---------------------------------------------------------
# New Year's Day Plot
# ---------------------------------------------------------

new_year_plot = pd.concat([
    new_years.assign(day="January 1"),
    jan2.assign(day="January 2")
])

new_year_plot["year"] = (
    new_year_plot["date"].dt.year
)

plt.figure(figsize=(10, 6))

for day, color in [
    ("January 1", "firebrick"),
    ("January 2", "steelblue")
]:

    subset = new_year_plot[
        new_year_plot["day"] == day
    ]

    plt.plot(
        subset["year"],
        subset["sales"],
        marker="o",
        linewidth=2,
        label=day,
        color=color
    )

plt.title(
    "Aggregate Sales on January 1 and January 2"
)

plt.xlabel("Year")
plt.ylabel("Total Sales")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

save_figure(
    "store_sales_new_year_effect.png"
)


# ---------------------------------------------------------
# Store Sales: Time-Based Validation Split
# ---------------------------------------------------------

VALIDATION_DAYS = 16

validation_start = (
    daily_sales["date"].max()
    -
    pd.Timedelta(days=VALIDATION_DAYS - 1)
)

daily_train = daily_sales[
    daily_sales["date"] < validation_start
].copy()

daily_validation = daily_sales[
    daily_sales["date"] >= validation_start
].copy()

print("\nTime-series validation split")
print("--------------------------------")

print(
    "Training start:",
    daily_train["date"].min()
)

print(
    "Training end:",
    daily_train["date"].max()
)

print(
    "Validation start:",
    daily_validation["date"].min()
)

print(
    "Validation end:",
    daily_validation["date"].max()
)

print(
    "Training observations:",
    len(daily_train)
)

print(
    "Validation observations:",
    len(daily_validation)
)

print("\nValidation observations:")
print(daily_validation)


# ---------------------------------------------------------
# Save: Validation Split
# ---------------------------------------------------------

validation_summary = pd.DataFrame({
    "dataset": [
        "model_training",
        "validation"
    ],
    "start_date": [
        daily_train["date"].min(),
        daily_validation["date"].min()
    ],
    "end_date": [
        daily_train["date"].max(),
        daily_validation["date"].max()
    ],
    "observations": [
        len(daily_train),
        len(daily_validation)
    ]
})

validation_summary.to_csv(
    TABLES_DIR / "store_sales_validation_summary.csv",
    index=False
)

daily_validation.to_csv(
    TABLES_DIR / "store_sales_validation_observations.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_validation_summary.csv"
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_validation_observations.csv"
)


# ---------------------------------------------------------
# Validation Period Plot
# ---------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    daily_train["date"],
    daily_train["sales"],
    color="steelblue",
    linewidth=0.8,
    label="Model Training"
)

plt.plot(
    daily_validation["date"],
    daily_validation["sales"],
    color="darkorange",
    linewidth=2,
    label="Validation Period"
)

plt.axvline(
    validation_start,
    color="black",
    linestyle="--",
    linewidth=1
)

plt.title(
    "Store Sales: Time-Series Training and Validation Split"
)

plt.xlabel("Date")
plt.ylabel("Total Sales")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

save_figure(
    "store_sales_validation_split.png"
)


# ---------------------------------------------------------
# Weekly Seasonality Check
# ---------------------------------------------------------

# Create day-of-week variable only for analysis.
daily_sales["day_of_week"] = (
    daily_sales["date"].dt.day_name()
)

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_summary = (
    daily_sales
    .groupby("day_of_week")["sales"]
    .agg([
        "mean",
        "median",
        "std"
    ])
    .reindex(weekday_order)
)

print("\nSales by day of week:")
print(weekday_summary)


# ---------------------------------------------------------
# Save: Weekly Seasonality Summary
# ---------------------------------------------------------

weekday_summary.reset_index().to_csv(
    TABLES_DIR / "store_sales_weekday_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_weekday_summary.csv"
)


# ---------------------------------------------------------
# Weekly Seasonality Plot
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

weekday_summary["mean"].plot(
    kind="bar",
    color="steelblue"
)

plt.title(
    "Average Daily Sales by Day of Week"
)

plt.xlabel("Day of Week")
plt.ylabel("Average Sales")

plt.xticks(rotation=45)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

save_figure(
    "store_sales_weekly_seasonality.png"
)


# ---------------------------------------------------------
# Weekly Seasonality: Relative Difference
# ---------------------------------------------------------

weekday_mean = weekday_summary["mean"]

weekday_relative = (
    weekday_mean / weekday_mean.mean()
)

weekday_relative_table = (
    weekday_relative
    .reset_index()
)

weekday_relative_table.columns = [
    "day_of_week",
    "relative_to_weekly_mean"
]

weekday_relative_table.to_csv(
    TABLES_DIR / "store_sales_weekday_relative.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR / "store_sales_weekday_relative.csv"
)


# ---------------------------------------------------------
# Create Durable Inspection Results
# ---------------------------------------------------------

inspection_results = {
    "project": "Assignment 8",
    "dataset": "Store Sales Time Series Forecasting",

    "training": {
        "rows": int(store_train.shape[0]),
        "columns": int(store_train.shape[1]),
        "start_date": str(
            store_train["date"].min().date()
        ),
        "end_date": str(
            store_train["date"].max().date()
        ),
        "number_of_stores": int(
            store_train["store_nbr"].nunique()
        ),
        "number_of_product_families": int(
            store_train["family"].nunique()
        ),
        "store_family_combinations": int(
            len(train_combinations)
        ),
        "missing_values": int(
            store_train.isna().sum().sum()
        )
    },

    "test": {
        "rows": int(store_test.shape[0]),
        "columns": int(store_test.shape[1]),
        "start_date": str(
            store_test["date"].min().date()
        ),
        "end_date": str(
            store_test["date"].max().date()
        ),
        "number_of_stores": int(
            store_test["store_nbr"].nunique()
        ),
        "number_of_product_families": int(
            store_test["family"].nunique()
        ),
        "store_family_combinations": int(
            len(test_combinations)
        )
    },

    "calendar": {
        "training_expected_days": int(
            len(train_dates)
        ),
        "training_observed_days": int(
            store_train["date"].nunique()
        ),
        "training_missing_days": int(
            len(missing_train_dates)
        ),
        "test_expected_days": int(
            len(test_dates)
        ),
        "test_observed_days": int(
            store_test["date"].nunique()
        ),
        "test_missing_days": int(
            len(missing_test_dates)
        )
    },

    "holiday_analysis": {
        "holiday_rows": int(len(holidays)),
        "christmas_observations": int(
            len(christmas)
        ),
        "missing_training_dates_are_christmas": bool(
            set(missing_train_dates)
            .issubset(set(christmas["date"]))
        )
    },

    "validation": {
        "horizon_days": int(VALIDATION_DAYS),
        "training_start": str(
            daily_train["date"].min().date()
        ),
        "training_end": str(
            daily_train["date"].max().date()
        ),
        "validation_start": str(
            daily_validation["date"].min().date()
        ),
        "validation_end": str(
            daily_validation["date"].max().date()
        ),
        "training_observations": int(
            len(daily_train)
        ),
        "validation_observations": int(
            len(daily_validation)
        )
    }
}


with open(
    RESULTS_DIR / "store_sales_inspection_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        inspection_results,
        f,
        indent=4
    )


print(
    "\nSaved results:",
    RESULTS_DIR / "store_sales_inspection_results.json"
)


# ---------------------------------------------------------
# Final Artifact Inventory
# ---------------------------------------------------------

print("\n============================================================")
print("01_data_inspection.py completed successfully")
print("============================================================")

print("\nFigures saved:")
for file in sorted(FIGURES_DIR.glob("store_sales_*.png")):
    print(" ", file.name)

print("\nTables saved:")
for file in sorted(TABLES_DIR.glob("store_sales_*.csv")):
    print(" ", file.name)

print("\nResults saved:")
for file in sorted(RESULTS_DIR.glob("store_sales_*.json")):
    print(" ", file.name)

print("\n01_data_inspection.py completed successfully.")
