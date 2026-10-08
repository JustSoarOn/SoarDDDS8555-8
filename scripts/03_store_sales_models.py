# ============================================================
# Assignment 8: Build & Evaluate
# Time Series, Regression, and Classification Models
#
# 03_store_sales_models.py
#
# Purpose:
#   Build and evaluate forecasting models for the Store Sales
#   time-series dataset and save durable, report-ready
#   artifacts for SoarDDDS8555-8-Report.Rmd.
#
# Modeling unit:
#   Aggregate daily sales across ALL stores and product
#   families.
#
# Models:
#   1. Naive
#   2. Seasonal Naive (7-day lag)
#   3. 7-Day Moving Average
#   4. Linear Trend
#   5. Day-of-Week Mean
#
# Validation period:
#   2017-07-31 through 2017-08-15
#
# Test period:
#   2017-08-16 through 2017-08-31
#
# IMPORTANT:
#   - The test set is never used for model selection.
#   - Model selection is based on validation RMSE.
#   - This script calculates and saves factual results.
#   - Interpretation and report presentation are handled by
#     SoarDDDS8555-8-Report.Rmd.
# ============================================================


from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Project directories
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

STORE_DIR = (
    DATA_DIR
    / "store_sales_time_series_forecasting"
)


# ---------------------------------------------------------
# Output directories
# ---------------------------------------------------------

OUTPUT_DIR = PROJECT_DIR / "output"

FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = OUTPUT_DIR / "results"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUBMISSIONS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------

VALIDATION_START = pd.Timestamp(
    "2017-07-31"
)

VALIDATION_END = pd.Timestamp(
    "2017-08-15"
)

TEST_START = pd.Timestamp(
    "2017-08-16"
)

TEST_END = pd.Timestamp(
    "2017-08-31"
)

SEASONAL_LAG = 7

MOVING_AVERAGE_WINDOW = 7

MODEL_SELECTION_CRITERION = "RMSE"


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def safe_mape(actual, predicted):
    """
    Calculate MAPE while excluding observations where the
    actual value is zero.
    """

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    nonzero = actual != 0

    if not np.any(nonzero):
        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    actual[nonzero]
                    - predicted[nonzero]
                )
                / actual[nonzero]
            )
        )
        * 100
    )


def smape(actual, predicted):
    """
    Calculate symmetric MAPE.

    If both actual and predicted are zero, the observation
    contributes zero to the final metric.
    """

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    numerator = (
        2
        * np.abs(actual - predicted)
    )

    result = np.zeros_like(
        denominator
    )

    nonzero = denominator != 0

    result[nonzero] = (
        numerator[nonzero]
        / denominator[nonzero]
    )

    return np.mean(result) * 100


def evaluate_forecast(actual, predicted):
    """
    Calculate standard forecasting metrics.
    """

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    error = (
        actual
        - predicted
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    mape = safe_mape(
        actual,
        predicted
    )

    smape_value = smape(
        actual,
        predicted
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_percent": (
            float(mape)
            if not np.isnan(mape)
            else None
        ),
        "sMAPE_percent": float(
            smape_value
        )
    }


# ---------------------------------------------------------
# Start
# ---------------------------------------------------------

print("=" * 60)
print("03_store_sales_models.py")
print("=" * 60)


# ---------------------------------------------------------
# Load training data
# ---------------------------------------------------------

print("\nLoading Store Sales training data...")

store_train = pd.read_csv(
    STORE_DIR / "train.csv",
    parse_dates=["date"]
)

print(
    "Training data shape:",
    store_train.shape
)

print(
    "Date range:",
    store_train["date"].min(),
    "to",
    store_train["date"].max()
)


# ---------------------------------------------------------
# Create aggregate daily sales series
# ---------------------------------------------------------

daily_sales = (
    store_train
    .groupby(
        "date",
        as_index=False
    )["sales"]
    .sum()
    .sort_values("date")
    .reset_index(drop=True)
)

print(
    "\nDaily observations:",
    len(daily_sales)
)


# ---------------------------------------------------------
# Validate calendar structure
# ---------------------------------------------------------

calendar = pd.date_range(
    daily_sales["date"].min(),
    daily_sales["date"].max(),
    freq="D"
)

missing_dates = calendar.difference(
    daily_sales["date"]
)

print(
    "Expected calendar days:",
    len(calendar)
)

print(
    "Observed calendar days:",
    len(daily_sales)
)

print(
    "Missing calendar days:",
    len(missing_dates)
)

print(
    "Missing dates:"
)

for date in missing_dates:
    print(
        " ",
        date.date()
    )


# ---------------------------------------------------------
# Save calendar validation artifacts
# ---------------------------------------------------------

calendar_summary = pd.DataFrame({
    "metric": [
        "expected_calendar_days",
        "observed_calendar_days",
        "missing_calendar_days"
    ],
    "value": [
        len(calendar),
        len(daily_sales),
        len(missing_dates)
    ]
})

calendar_summary.to_csv(
    TABLES_DIR
    / "store_sales_model_calendar_summary.csv",
    index=False
)

missing_calendar_dates = pd.DataFrame({
    "date": missing_dates
})

missing_calendar_dates.to_csv(
    TABLES_DIR
    / "store_sales_model_missing_calendar_dates.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_model_calendar_summary.csv"
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_model_missing_calendar_dates.csv"
)


# ---------------------------------------------------------
# Add time features
# ---------------------------------------------------------

daily_sales["year"] = (
    daily_sales["date"].dt.year
)

daily_sales["month"] = (
    daily_sales["date"].dt.month
)

daily_sales["day_of_week"] = (
    daily_sales["date"].dt.dayofweek
)

daily_sales["day_of_week_name"] = (
    daily_sales["date"].dt.day_name()
)

daily_sales["day_of_year"] = (
    daily_sales["date"].dt.dayofyear
)

daily_sales["time_index"] = (
    np.arange(
        len(daily_sales)
    )
)


# ---------------------------------------------------------
# Save aggregate modeling series
# ---------------------------------------------------------

daily_sales[
    [
        "date",
        "sales",
        "year",
        "month",
        "day_of_week",
        "day_of_week_name",
        "day_of_year",
        "time_index"
    ]
].to_csv(
    TABLES_DIR
    / "store_sales_model_daily_series.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_model_daily_series.csv"
)


# ---------------------------------------------------------
# Validation split
# ---------------------------------------------------------

train_ts = daily_sales[
    daily_sales["date"]
    < VALIDATION_START
].copy()

validation_ts = daily_sales[
    (
        daily_sales["date"]
        >= VALIDATION_START
    )
    &
    (
        daily_sales["date"]
        <= VALIDATION_END
    )
].copy()


print("\nTime-series validation split")
print("-" * 40)

print(
    "Training start:",
    train_ts["date"].min()
)

print(
    "Training end:",
    train_ts["date"].max()
)

print(
    "Validation start:",
    validation_ts["date"].min()
)

print(
    "Validation end:",
    validation_ts["date"].max()
)

print(
    "Training observations:",
    len(train_ts)
)

print(
    "Validation observations:",
    len(validation_ts)
)


# ---------------------------------------------------------
# Validate expected validation period
# ---------------------------------------------------------

expected_validation_dates = pd.date_range(
    VALIDATION_START,
    VALIDATION_END,
    freq="D"
)

actual_validation_dates = (
    validation_ts["date"]
    .sort_values()
    .reset_index(drop=True)
)

if not actual_validation_dates.equals(
    pd.Series(expected_validation_dates)
):
    raise ValueError(
        "Validation period does not contain the "
        "expected consecutive dates."
    )


# ---------------------------------------------------------
# Save validation information
# ---------------------------------------------------------

validation_summary = pd.DataFrame({
    "dataset": [
        "training",
        "validation"
    ],
    "start_date": [
        train_ts["date"].min(),
        validation_ts["date"].min()
    ],
    "end_date": [
        train_ts["date"].max(),
        validation_ts["date"].max()
    ],
    "observations": [
        len(train_ts),
        len(validation_ts)
    ]
})

validation_summary.to_csv(
    TABLES_DIR
    / "store_sales_model_validation_summary.csv",
    index=False
)

validation_ts[
    [
        "date",
        "sales",
        "day_of_week",
        "day_of_week_name",
        "time_index"
    ]
].to_csv(
    TABLES_DIR
    / "store_sales_model_validation_observations.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_model_validation_summary.csv"
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_model_validation_observations.csv"
)


# ---------------------------------------------------------
# Actual validation values
# ---------------------------------------------------------

actual = (
    validation_ts["sales"]
    .to_numpy()
)


# =========================================================
# MODEL 1
# Naive Forecast
# =========================================================

naive_value = (
    train_ts["sales"].iloc[-1]
)

naive_prediction = np.repeat(
    naive_value,
    len(validation_ts)
)


# =========================================================
# MODEL 2
# Seasonal Naive Forecast
# =========================================================

seasonal_history = (
    train_ts
    .set_index("date")["sales"]
)

seasonal_naive_prediction = []

for date in validation_ts["date"]:

    previous_week = (
        date
        - pd.Timedelta(
            days=SEASONAL_LAG
        )
    )

    if previous_week in seasonal_history.index:

        value = (
            seasonal_history.loc[
                previous_week
            ]
        )

    else:

        value = naive_value

    seasonal_naive_prediction.append(
        value
    )

seasonal_naive_prediction = np.asarray(
    seasonal_naive_prediction,
    dtype=float
)


# =========================================================
# MODEL 3
# 7-Day Moving Average
# =========================================================

moving_average_value = (
    train_ts["sales"]
    .tail(
        MOVING_AVERAGE_WINDOW
    )
    .mean()
)

moving_average_prediction = np.repeat(
    moving_average_value,
    len(validation_ts)
)


# =========================================================
# MODEL 4
# Linear Trend
# =========================================================

x_train = (
    train_ts["time_index"]
    .to_numpy()
)

y_train = (
    train_ts["sales"]
    .to_numpy()
)

trend_coefficients = np.polyfit(
    x_train,
    y_train,
    deg=1
)

trend_model = np.poly1d(
    trend_coefficients
)

x_validation = (
    validation_ts["time_index"]
    .to_numpy()
)

trend_prediction = trend_model(
    x_validation
)


# =========================================================
# MODEL 5
# Day-of-Week Mean
# =========================================================

weekday_means = (
    train_ts
    .groupby(
        "day_of_week"
    )["sales"]
    .mean()
)

weekday_prediction = (
    validation_ts["day_of_week"]
    .map(
        weekday_means
    )
    .to_numpy()
)


# ---------------------------------------------------------
# Candidate forecast dictionary
# ---------------------------------------------------------

forecasts = {
    "Naive": naive_prediction,

    "Seasonal Naive":
        seasonal_naive_prediction,

    "7-Day Moving Average":
        moving_average_prediction,

    "Linear Trend":
        trend_prediction,

    "Day-of-Week Mean":
        weekday_prediction
}


# ---------------------------------------------------------
# Validate forecast lengths
# ---------------------------------------------------------

for model_name, prediction in forecasts.items():

    if len(prediction) != len(actual):

        raise ValueError(
            f"{model_name} forecast length "
            "does not match validation length."
        )


# ---------------------------------------------------------
# Completion marker
# ---------------------------------------------------------

print(
    "\ncalculations completed."
)

print(
    "Candidate models:",
    list(forecasts.keys())
)

print(
    "Validation observations:",
    len(validation_ts)
)

# =========================================================
# Model Evaluation, Validation Artifacts, Plots,
# and New Year's Day Analysis
# =========================================================


# ---------------------------------------------------------
# Evaluate candidate models
# ---------------------------------------------------------

evaluation_rows = []

for model_name, prediction in forecasts.items():

    metrics = evaluate_forecast(
        actual,
        prediction
    )

    evaluation_rows.append({
        "model": model_name,
        **metrics
    })


model_evaluation = pd.DataFrame(
    evaluation_rows
)


# ---------------------------------------------------------
# Preserve a stable model ordering
# ---------------------------------------------------------

model_order = [
    "Naive",
    "Seasonal Naive",
    "7-Day Moving Average",
    "Linear Trend",
    "Day-of-Week Mean"
]

model_evaluation["model"] = pd.Categorical(
    model_evaluation["model"],
    categories=model_order,
    ordered=True
)

model_evaluation = (
    model_evaluation
    .sort_values("model")
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# Save model evaluation
# ---------------------------------------------------------

model_evaluation.to_csv(
    TABLES_DIR
    / "store_sales_model_evaluation.csv",
    index=False
)

print(
    "\nSaved table:",
    TABLES_DIR
    / "store_sales_model_evaluation.csv"
)

print("\nModel evaluation:")
print(
    model_evaluation.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Create validation forecast comparison table
# ---------------------------------------------------------

forecast_comparison = validation_ts[
    [
        "date",
        "sales",
        "day_of_week",
        "day_of_week_name"
    ]
].copy()

forecast_comparison.rename(
    columns={
        "sales": "actual_sales"
    },
    inplace=True
)


# ---------------------------------------------------------
# Add every model's validation forecast
# ---------------------------------------------------------

forecast_column_map = {
    "Naive":
        "naive_forecast",

    "Seasonal Naive":
        "seasonal_naive_forecast",

    "7-Day Moving Average":
        "moving_average_forecast",

    "Linear Trend":
        "linear_trend_forecast",

    "Day-of-Week Mean":
        "day_of_week_mean_forecast"
}


for model_name, prediction in forecasts.items():

    forecast_column = (
        forecast_column_map[
            model_name
        ]
    )

    forecast_comparison[
        forecast_column
    ] = prediction


# ---------------------------------------------------------
# Save validation forecasts
# ---------------------------------------------------------

forecast_comparison.to_csv(
    TABLES_DIR
    / "store_sales_validation_forecasts.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_validation_forecasts.csv"
)


# ---------------------------------------------------------
# Save validation errors
# ---------------------------------------------------------

validation_errors = validation_ts[
    [
        "date",
        "sales",
        "day_of_week",
        "day_of_week_name"
    ]
].copy()

validation_errors.rename(
    columns={
        "sales": "actual_sales"
    },
    inplace=True
)


for model_name, prediction in forecasts.items():

    forecast_column = (
        forecast_column_map[
            model_name
        ]
    )

    error_column = (
        model_name
        .lower()
        .replace("-", "")
        .replace(" ", "_")
        + "_error"
    )

    validation_errors[
        error_column
    ] = (
        validation_errors["actual_sales"]
        - prediction
    )


validation_errors.to_csv(
    TABLES_DIR
    / "store_sales_validation_errors.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_validation_errors.csv"
)


# ---------------------------------------------------------
# Identify validation performance ordering
# ---------------------------------------------------------

validation_ranking = (
    model_evaluation
    .sort_values(
        MODEL_SELECTION_CRITERION,
        ascending=True
    )
    .reset_index(drop=True)
)

validation_ranking[
    "validation_rank"
] = np.arange(
    1,
    len(validation_ranking) + 1
)


validation_ranking.to_csv(
    TABLES_DIR
    / "store_sales_model_validation_ranking.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_model_validation_ranking.csv"
)


# ---------------------------------------------------------
# Forecast comparison figure
# ---------------------------------------------------------

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    validation_ts["date"],
    actual,
    color="black",
    linewidth=2.5,
    marker="o",
    label="Actual"
)


colors = {
    "Naive": "steelblue",
    "Seasonal Naive": "darkorange",
    "7-Day Moving Average": "forestgreen",
    "Linear Trend": "firebrick",
    "Day-of-Week Mean": "purple"
}


for model_name, prediction in forecasts.items():

    plt.plot(
        validation_ts["date"],
        prediction,
        linewidth=1.8,
        marker=".",
        color=colors[model_name],
        label=model_name
    )


plt.title(
    "Store Sales Validation Forecast Comparison"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Total Daily Sales"
)

plt.xticks(
    rotation=45
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "store_sales_model_forecast_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR
    / "store_sales_model_forecast_comparison.png"
)


# ---------------------------------------------------------
# Validation metrics comparison figure
# ---------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)

plot_data = (
    model_evaluation
    .sort_values(
        "RMSE",
        ascending=True
    )
)

plt.barh(
    plot_data["model"],
    plot_data["RMSE"],
    color="steelblue"
)

plt.xlabel(
    "Validation RMSE"
)

plt.ylabel(
    "Model"
)

plt.title(
    "Validation RMSE by Forecasting Model"
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "store_sales_model_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR
    / "store_sales_model_rmse_comparison.png"
)


# =========================================================
# New Year's Day Analysis
# =========================================================

print(
    "\nNew Year's Day analysis"
)


# ---------------------------------------------------------
# Identify January 1 observations
# ---------------------------------------------------------

new_year_dates = daily_sales[
    (
        daily_sales["month"] == 1
    )
    &
    (
        daily_sales["date"].dt.day == 1
    )
][
    [
        "date",
        "sales"
    ]
].copy()


# ---------------------------------------------------------
# Identify January 2 observations
# ---------------------------------------------------------

january_2_dates = daily_sales[
    (
        daily_sales["month"] == 1
    )
    &
    (
        daily_sales["date"].dt.day == 2
    )
][
    [
        "date",
        "sales"
    ]
].copy()


# ---------------------------------------------------------
# Match January 1 to the following January 2
# ---------------------------------------------------------

new_year_comparison = (
    new_year_dates
    .assign(
        january_2_date=lambda x:
            x["date"]
            + pd.Timedelta(days=1)
    )
    .merge(
        january_2_dates.rename(
            columns={
                "date":
                    "january_2_date",

                "sales":
                    "january_2_sales"
            }
        ),
        on="january_2_date",
        how="left"
    )
)


new_year_comparison.rename(
    columns={
        "date":
            "january_1_date",

        "sales":
            "january_1_sales"
    },
    inplace=True
)


# ---------------------------------------------------------
# Calculate January 1 vs. January 2 difference
# ---------------------------------------------------------

new_year_comparison[
    "sales_difference_jan1_minus_jan2"
] = (
    new_year_comparison[
        "january_1_sales"
    ]
    -
    new_year_comparison[
        "january_2_sales"
    ]
)


# ---------------------------------------------------------
# Calculate January 1 percentage difference
# relative to January 2
# ---------------------------------------------------------

new_year_comparison[
    "sales_percent_difference_vs_jan2"
] = np.where(
    new_year_comparison[
        "january_2_sales"
    ].notna()
    &
    (
        new_year_comparison[
            "january_2_sales"
        ]
        != 0
    ),
    (
        (
            new_year_comparison[
                "january_1_sales"
            ]
            -
            new_year_comparison[
                "january_2_sales"
            ]
        )
        /
        new_year_comparison[
            "january_2_sales"
        ]
        * 100
    ),
    np.nan
)


# ---------------------------------------------------------
# Explicit denominator flag
# ---------------------------------------------------------

new_year_comparison[
    "january_2_zero_sales"
] = (
    new_year_comparison[
        "january_2_sales"
    ]
    == 0
)


# ---------------------------------------------------------
# Explicit missing-pair flag
# ---------------------------------------------------------

new_year_comparison[
    "january_2_missing"
] = (
    new_year_comparison[
        "january_2_sales"
    ].isna()
)


# ---------------------------------------------------------
# Save New Year's Day analysis
# ---------------------------------------------------------

new_year_comparison.to_csv(
    TABLES_DIR
    / "store_sales_new_year_model_analysis.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_new_year_model_analysis.csv"
)


# ---------------------------------------------------------
# Save compact New Year's summary
# ---------------------------------------------------------

new_year_summary = pd.DataFrame({
    "metric": [
        "january_1_observations",
        "january_2_matched_observations",
        "january_2_missing_pairs",
        "january_2_zero_sales_cases"
    ],
    "value": [
        len(new_year_comparison),

        int(
            new_year_comparison[
                "january_2_sales"
            ].notna().sum()
        ),

        int(
            new_year_comparison[
                "january_2_missing"
            ].sum()
        ),

        int(
            new_year_comparison[
                "january_2_zero_sales"
            ].sum()
        )
    ]
})


new_year_summary.to_csv(
    TABLES_DIR
    / "store_sales_new_year_model_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_new_year_model_summary.csv"
)


# ---------------------------------------------------------
# New Year's Day comparison figure
# ---------------------------------------------------------

new_year_plot = pd.concat(
    [
        new_year_dates.assign(
            day="January 1"
        ),
        january_2_dates.assign(
            day="January 2"
        )
    ],
    ignore_index=True
)


new_year_plot["year"] = (
    new_year_plot["date"].dt.year
)


plt.figure(
    figsize=(10, 6)
)


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

plt.xlabel(
    "Year"
)

plt.ylabel(
    "Total Sales"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "store_sales_new_year_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR
    / "store_sales_new_year_model_comparison.png"
)


# ---------------------------------------------------------
# Save results metadata
# ---------------------------------------------------------

metadata_results = {

    "validation_start":
        str(VALIDATION_START.date()),

    "validation_end":
        str(VALIDATION_END.date()),

    "validation_observations":
        int(len(validation_ts)),

    "models_evaluated":
        list(forecasts.keys()),

    "model_selection_criterion":
        MODEL_SELECTION_CRITERION,

    "new_year_observations":
        int(len(new_year_comparison)),

    "new_year_matched_january_2_observations":
        int(
            new_year_comparison[
                "january_2_sales"
            ].notna().sum()
        ),

    "new_year_missing_january_2_pairs":
        int(
            new_year_comparison[
                "january_2_missing"
            ].sum()
        ),

    "new_year_zero_january_2_cases":
        int(
            new_year_comparison[
                "january_2_zero_sales"
            ].sum()
        )
}


with open(
    RESULTS_DIR
    / "store_sales_model_validation_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata_results,
        file,
        indent=4
    )


print(
    "Saved results:",
    RESULTS_DIR
    / "store_sales_model_validation_results.json"
)


# ---------------------------------------------------------
# completion
# ---------------------------------------------------------

print(
    "\n completed."
)

print(
    "Validation evaluation, forecast tables, "
    "validation errors, comparison figures, "
    "and New Year's Day artifacts have been saved."
)



# =========================================================
# Test-Set Forecasting and Final Model Selection
# =========================================================

print("\nLoading Store Sales test data...")

store_sales_test_data = pd.read_csv(
    STORE_DIR / "test.csv",
    parse_dates=["date"]
)

print(
    "Test data shape:",
    store_sales_test_data.shape
)

print(
    "Test date range:",
    store_sales_test_data["date"].min(),
    "to",
    store_sales_test_data["date"].max()
)


# ---------------------------------------------------------
# Create unique test dates
# ---------------------------------------------------------

test_dates = (
    store_sales_test_data[
        ["date"]
    ]
    .drop_duplicates()
    .sort_values("date")
    .reset_index(drop=True)
)

print(
    "Test calendar observations:",
    len(test_dates)
)


# ---------------------------------------------------------
# Create complete historical daily sales series
# ---------------------------------------------------------

historical_daily_sales = (
    daily_sales[
        ["date", "sales"]
    ]
    .set_index("date")["sales"]
)


# ---------------------------------------------------------
# Calculate final training-window statistics
# ---------------------------------------------------------

final_moving_average_value = (
    historical_daily_sales
    .tail(MOVING_AVERAGE_WINDOW)
    .mean()
)


# ---------------------------------------------------------
# Seasonal Naive Test Forecast
# ---------------------------------------------------------

seasonal_naive_test_forecasts = []

for forecast_date in test_dates["date"]:

    previous_week_date = (
        forecast_date
        - pd.Timedelta(days=SEASONAL_LAG)
    )

    if previous_week_date in historical_daily_sales.index:

        forecast_value = (
            historical_daily_sales.loc[
                previous_week_date
            ]
        )

    else:

        forecast_value = (
            historical_daily_sales.iloc[-1]
        )

    seasonal_naive_test_forecasts.append(
        forecast_value
    )


seasonal_naive_test_forecasts = np.asarray(
    seasonal_naive_test_forecasts,
    dtype=float
)


# ---------------------------------------------------------
# Naive Test Forecast
# ---------------------------------------------------------

naive_test_forecasts = np.repeat(
    historical_daily_sales.iloc[-1],
    len(test_dates)
)


# ---------------------------------------------------------
# Moving Average Test Forecast
# ---------------------------------------------------------

moving_average_test_forecasts = np.repeat(
    final_moving_average_value,
    len(test_dates)
)


# ---------------------------------------------------------
# Linear Trend Test Forecast
# ---------------------------------------------------------

linear_trend_test_time_index = np.arange(
    len(daily_sales),
    len(daily_sales) + len(test_dates)
)

linear_trend_test_forecasts = trend_model(
    linear_trend_test_time_index
)


# ---------------------------------------------------------
# Day-of-Week Mean Test Forecast
# ---------------------------------------------------------

test_day_of_week_numbers = (
    test_dates["date"]
    .dt.dayofweek
)

day_of_week_mean_test_forecasts = (
    test_day_of_week_numbers
    .map(weekday_means)
    .to_numpy()
)


# ---------------------------------------------------------
# Store all candidate test forecasts
# ---------------------------------------------------------

test_forecast_candidates = {

    "Naive": naive_test_forecasts,

    "Seasonal Naive": (
        seasonal_naive_test_forecasts
    ),

    "7-Day Moving Average": (
        moving_average_test_forecasts
    ),

    "Linear Trend": (
        linear_trend_test_forecasts
    ),

    "Day-of-Week Mean": (
        day_of_week_mean_test_forecasts
    )
}


# ---------------------------------------------------------
# Create report-ready test forecast table
# ---------------------------------------------------------

test_forecast_comparison = pd.DataFrame({
    "date": test_dates["date"],
    "day_of_week": test_dates["date"].dt.day_name(),
    "naive_forecast": (
        naive_test_forecasts
    ),
    "seasonal_naive_forecast": (
        seasonal_naive_test_forecasts
    ),
    "moving_average_7_day_forecast": (
        moving_average_test_forecasts
    ),
    "linear_trend_forecast": (
        linear_trend_test_forecasts
    ),
    "day_of_week_mean_forecast": (
        day_of_week_mean_test_forecasts
    )
})


test_forecast_comparison.to_csv(
    TABLES_DIR
    / "store_sales_test_forecast_comparison.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_test_forecast_comparison.csv"
)


# ---------------------------------------------------------
# New Year's Day Test-Date Check
# ---------------------------------------------------------

test_new_year_dates = test_dates[
    (
        test_dates["date"].dt.month == 1
    )
    &
    (
        test_dates["date"].dt.day == 1
    )
].copy()

print(
    "New Year's Day observations in test:",
    len(test_new_year_dates)
)


# ---------------------------------------------------------
# Select Final Model Using Validation RMSE
# ---------------------------------------------------------

selected_model_row = (
    model_evaluation
    .sort_values(
        "RMSE",
        ascending=True
    )
    .iloc[0]
)

selected_forecasting_model = str(
    selected_model_row["model"]
)

selected_model_validation_rmse = float(
    selected_model_row["RMSE"]
)

selected_model_validation_mae = float(
    selected_model_row["MAE"]
)

selected_model_validation_mape = float(
    selected_model_row["MAPE_percent"]
)

selected_model_validation_smape = float(
    selected_model_row["sMAPE_percent"]
)


print(
    "\nSelected forecasting model:",
    selected_forecasting_model
)

print(
    "Model-selection criterion: Validation RMSE"
)

print(
    "Selected model validation RMSE:",
    selected_model_validation_rmse
)

print(
    "Selected model validation MAE:",
    selected_model_validation_mae
)


# ---------------------------------------------------------
# Map Selected Model to Test Forecast Column
# ---------------------------------------------------------

test_forecast_column_by_model = {

    "Naive":
        "naive_forecast",

    "Seasonal Naive":
        "seasonal_naive_forecast",

    "7-Day Moving Average":
        "moving_average_7_day_forecast",

    "Linear Trend":
        "linear_trend_forecast",

    "Day-of-Week Mean":
        "day_of_week_mean_forecast"
}


selected_test_forecast_column = (
    test_forecast_column_by_model[
        selected_forecasting_model
    ]
)


# ---------------------------------------------------------
# Create Primary Test Forecast
# ---------------------------------------------------------

primary_test_forecast = test_forecast_comparison[
    [
        "date",
        selected_test_forecast_column
    ]
].copy()

primary_test_forecast.rename(
    columns={
        selected_test_forecast_column:
            "forecast_sales"
    },
    inplace=True
)

primary_test_forecast[
    "selected_model"
] = selected_forecasting_model


primary_test_forecast.to_csv(
    TABLES_DIR
    / "store_sales_primary_test_forecast.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_primary_test_forecast.csv"
)


# ---------------------------------------------------------
# Primary Test Forecast Summary
# ---------------------------------------------------------

primary_test_forecast_summary = pd.DataFrame({

    "metric": [

        "selected_model",

        "model_selection_criterion",

        "validation_rmse",

        "validation_mae",

        "validation_mape_percent",

        "validation_smape_percent",

        "test_start",

        "test_end",

        "test_observations",

        "forecast_mean",

        "forecast_min",

        "forecast_max",

        "forecast_total"
    ],

    "value": [

        selected_forecasting_model,

        "Validation RMSE",

        selected_model_validation_rmse,

        selected_model_validation_mae,

        selected_model_validation_mape,

        selected_model_validation_smape,

        str(
            test_dates["date"].min().date()
        ),

        str(
            test_dates["date"].max().date()
        ),

        len(test_dates),

        float(
            primary_test_forecast[
                "forecast_sales"
            ].mean()
        ),

        float(
            primary_test_forecast[
                "forecast_sales"
            ].min()
        ),

        float(
            primary_test_forecast[
                "forecast_sales"
            ].max()
        ),

        float(
            primary_test_forecast[
                "forecast_sales"
            ].sum()
        )
    ]
})


primary_test_forecast_summary.to_csv(
    TABLES_DIR
    / "store_sales_primary_test_forecast_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_primary_test_forecast_summary.csv"
)


# ---------------------------------------------------------
# Final Model Results JSON
# ---------------------------------------------------------

model_results = {

    "project": "Assignment 8",

    "dataset":
        "Store Sales Time Series Forecasting",

    "validation": {

        "start_date":
            str(VALIDATION_START.date()),

        "end_date":
            str(VALIDATION_END.date()),

        "observations":
            int(len(validation_ts)),

        "model_selection_criterion":
            "Validation RMSE"
    },

    "test": {

        "start_date":
            str(TEST_START.date()),

        "end_date":
            str(TEST_END.date()),

        "observations":
            int(len(test_dates))
    },

    "candidate_models":
        list(model_evaluation["model"]),

    "selected_model":
        selected_forecasting_model,

    "selected_model_validation_metrics": {

        "RMSE":
            selected_model_validation_rmse,

        "MAE":
            selected_model_validation_mae,

        "MAPE_percent":
            selected_model_validation_mape,

        "sMAPE_percent":
            selected_model_validation_smape
    },

    "forecast_settings": {

        "seasonal_lag_days":
            int(SEASONAL_LAG),

        "moving_average_window_days":
            int(MOVING_AVERAGE_WINDOW)
    },

    "new_year_analysis": {

        "training_comparison_observations":
            int(len(new_year_comparison)),

        "zero_denominator_cases":
            int(
                new_year_comparison[
                    "january_2_zero_sales"
                ].sum()
            ),

        "test_new_year_observations":
            int(len(test_new_year_dates))
    },

    "primary_test_forecast": {

        "forecast_column":
            selected_test_forecast_column,

        "mean":
            float(
                primary_test_forecast[
                    "forecast_sales"
                ].mean()
            ),

        "minimum":
            float(
                primary_test_forecast[
                    "forecast_sales"
                ].min()
            ),

        "maximum":
            float(
                primary_test_forecast[
                    "forecast_sales"
                ].max()
            ),

        "total":
            float(
                primary_test_forecast[
                    "forecast_sales"
                ].sum()
            )
    }
}


with open(
    RESULTS_DIR
    / "store_sales_model_results.json",
    "w",
    encoding="utf-8"
) as results_file:

    json.dump(
        model_results,
        results_file,
        indent=4
    )


print(
    "Saved results:",
    RESULTS_DIR
    / "store_sales_model_results.json"
)


# ---------------------------------------------------------
# Final Forecast Artifact Inventory
# ---------------------------------------------------------

print("\nFinal Store Sales forecasting artifacts:")

print(
    "  Selected model:",
    selected_forecasting_model
)

print(
    "  Selection criterion:",
    "Validation RMSE"
)

print(
    "  Primary forecast rows:",
    len(primary_test_forecast)
)

print(
    "  Primary forecast column:",
    selected_test_forecast_column
)

print("\n" + "=" * 60)
print(
    "\n03_store_sales_models.py completed successfully."
)
print("=" * 60)
