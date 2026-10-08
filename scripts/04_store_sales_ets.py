# ============================================================
# Assignment 8: Build & Evaluate
# Time Series, Regression, and Classification Models
#
# 04_store_sales_ets.py
#
# Purpose:
#   Build and evaluate an Exponential Smoothing / ETS-style
#   forecasting model for the Store Sales time-series dataset.
#
# IMPORTANT:
#   - This script is independently executable.
#   - It does NOT import or depend on 03_store_sales_models.py.
#   - It creates its own aggregate daily sales series.
#   - It explicitly preserves the date column.
#   - It validates and repairs the daily calendar before
#     fitting the forecasting model.
#   - The test set is never used for model selection.
#   - Model selection is based on validation RMSE.
#   - Durable artifacts are saved for Report.Rmd.
#   - Report.Rmd can consume these artifacts without
#     re-running this time-consuming model.
#
# Modeling unit:
#   Aggregate daily sales across ALL stores and product
#   families.
#
# Validation period:
#   2017-07-31 through 2017-08-15
#
# Test period:
#   2017-08-16 through 2017-08-31
#
# Model:
#   Exponential Smoothing / ETS-style model using
#   statsmodels ExponentialSmoothing.
# ============================================================


from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ---------------------------------------------------------
# Suppress non-critical warnings
# ---------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning
)


# ---------------------------------------------------------
# Project directories
# ---------------------------------------------------------

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

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

SEASONAL_PERIODS = 7

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
        denominator,
        dtype=float
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
print("04_store_sales_ets.py")
print("=" * 60)


# ---------------------------------------------------------
# Load Store Sales training data
# ---------------------------------------------------------

print(
    "\nLoading Store Sales training data..."
)

store_train = pd.read_csv(
    STORE_DIR / "train.csv",
    parse_dates=["date"]
)

print(
    "Training data shape:",
    store_train.shape
)

print(
    "Training date range:",
    store_train["date"].min(),
    "to",
    store_train["date"].max()
)


# ---------------------------------------------------------
# Validate required columns
# ---------------------------------------------------------

required_columns = [
    "date",
    "sales"
]

missing_columns = [
    column
    for column in required_columns
    if column not in store_train.columns
]

if missing_columns:

    raise ValueError(
        "Store Sales training data is missing "
        f"required columns: {missing_columns}"
    )


# ---------------------------------------------------------
# Create aggregate daily sales series
#
# IMPORTANT:
#   The date column is explicitly retained here.
# ---------------------------------------------------------

daily_sales = (
    store_train[
        ["date", "sales"]
    ]
    .groupby(
        "date",
        as_index=False
    )["sales"]
    .sum()
    .sort_values(
        "date"
    )
    .reset_index(
        drop=True
    )
)


# ---------------------------------------------------------
# Confirm date column exists
# ---------------------------------------------------------

if "date" not in daily_sales.columns:

    raise ValueError(
        "The aggregate daily sales series does not "
        "contain the required 'date' column."
    )


if "sales" not in daily_sales.columns:

    raise ValueError(
        "The aggregate daily sales series does not "
        "contain the required 'sales' column."
    )


print(
    "\nAggregate daily sales created successfully."
)

print(
    "Daily observations before calendar completion:",
    len(daily_sales)
)

print(
    "Daily sales columns:",
    list(daily_sales.columns)
)

print(
    "Daily date range:",
    daily_sales["date"].min(),
    "to",
    daily_sales["date"].max()
)


# ---------------------------------------------------------
# Validate duplicate dates
# ---------------------------------------------------------

duplicate_date_count = int(
    daily_sales["date"]
    .duplicated()
    .sum()
)

if duplicate_date_count != 0:

    raise ValueError(
        "Aggregate daily sales contains duplicate "
        f"dates: {duplicate_date_count}"
    )


# ---------------------------------------------------------
# Validate calendar structure
# ---------------------------------------------------------

expected_calendar = pd.date_range(
    start=daily_sales["date"].min(),
    end=daily_sales["date"].max(),
    freq="D"
)

observed_dates = pd.DatetimeIndex(
    daily_sales["date"]
)

missing_dates = (
    expected_calendar
    .difference(observed_dates)
)


print(
    "\nCalendar validation:"
)

print(
    "Expected calendar days:",
    len(expected_calendar)
)

print(
    "Observed calendar days:",
    len(observed_dates)
)

print(
    "Missing calendar days:",
    len(missing_dates)
)


# ---------------------------------------------------------
# Save calendar validation artifacts
# ---------------------------------------------------------

calendar_summary = pd.DataFrame({
    "metric": [
        "expected_calendar_days",
        "observed_calendar_days",
        "missing_calendar_days",
        "duplicate_date_count"
    ],
    "value": [
        len(expected_calendar),
        len(observed_dates),
        len(missing_dates),
        duplicate_date_count
    ]
})


calendar_summary.to_csv(
    TABLES_DIR
    / "store_sales_ets_calendar_summary.csv",
    index=False
)


missing_calendar_dates = pd.DataFrame({
    "date": missing_dates
})


missing_calendar_dates.to_csv(
    TABLES_DIR
    / "store_sales_ets_missing_calendar_dates.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_ets_calendar_summary.csv"
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_ets_missing_calendar_dates.csv"
)


# ---------------------------------------------------------
# Complete the daily calendar
#
# Missing dates are filled with zero aggregate sales.
#
# This is an explicit data-treatment decision and is saved
# as an artifact for transparent reporting.
# ---------------------------------------------------------

daily_sales_complete = (
    daily_sales
    .set_index("date")
    .reindex(expected_calendar)
    .rename_axis("date")
    .reset_index()
)


daily_sales_complete[
    "sales_was_missing"
] = (
    daily_sales_complete["sales"]
    .isna()
)


missing_sales_count = int(
    daily_sales_complete[
        "sales_was_missing"
    ].sum()
)


daily_sales_complete[
    "sales"
] = (
    daily_sales_complete["sales"]
    .fillna(0.0)
)


print(
    "\nCalendar completion:"
)

print(
    "Missing daily observations identified:",
    missing_sales_count
)

print(
    "Missing daily observations filled with zero:",
    missing_sales_count
)


# ---------------------------------------------------------
# Add time features
# ---------------------------------------------------------

daily_sales_complete["year"] = (
    daily_sales_complete["date"].dt.year
)

daily_sales_complete["month"] = (
    daily_sales_complete["date"].dt.month
)

daily_sales_complete["day_of_week"] = (
    daily_sales_complete["date"].dt.dayofweek
)

daily_sales_complete["day_of_week_name"] = (
    daily_sales_complete[
        "date"
    ].dt.day_name()
)

daily_sales_complete["day_of_year"] = (
    daily_sales_complete["date"].dt.dayofyear
)

daily_sales_complete["time_index"] = (
    np.arange(
        len(daily_sales_complete)
    )
)


# ---------------------------------------------------------
# Save complete aggregate modeling series
# ---------------------------------------------------------

daily_sales_complete[
    [
        "date",
        "sales",
        "sales_was_missing",
        "year",
        "month",
        "day_of_week",
        "day_of_week_name",
        "day_of_year",
        "time_index"
    ]
].to_csv(
    TABLES_DIR
    / "store_sales_ets_daily_series.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_ets_daily_series.csv"
)


# ---------------------------------------------------------
# Verify continuous daily series after completion
# ---------------------------------------------------------

complete_calendar_check = pd.date_range(
    daily_sales_complete["date"].min(),
    daily_sales_complete["date"].max(),
    freq="D"
)


if not pd.DatetimeIndex(
    daily_sales_complete["date"]
).equals(
    complete_calendar_check
):

    raise ValueError(
        "Calendar completion failed. The ETS input "
        "series is not continuous."
    )


print(
    "Continuous daily ETS series verified."
)


# ---------------------------------------------------------
# Create time-series training and validation data
# ---------------------------------------------------------

train_ts = daily_sales_complete[
    daily_sales_complete["date"]
    < VALIDATION_START
].copy()


validation_ts = daily_sales_complete[
    (
        daily_sales_complete["date"]
        >= VALIDATION_START
    )
    &
    (
        daily_sales_complete["date"]
        <= VALIDATION_END
    )
].copy()


print(
    "\nTime-series validation split"
)

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


actual_validation_dates = pd.DatetimeIndex(
    validation_ts["date"]
)


if not actual_validation_dates.equals(
    expected_validation_dates
):

    raise ValueError(
        "Validation period does not contain the "
        "expected consecutive dates."
    )


# ---------------------------------------------------------
# Save validation summary
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
    / "store_sales_ets_validation_summary.csv",
    index=False
)


validation_ts[
    [
        "date",
        "sales",
        "sales_was_missing",
        "day_of_week",
        "day_of_week_name",
        "time_index"
    ]
].to_csv(
    TABLES_DIR
    / "store_sales_ets_validation_observations.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_ets_validation_summary.csv"
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_ets_validation_observations.csv"
)


# ---------------------------------------------------------
# Prepare ETS training and validation arrays
# ---------------------------------------------------------

ets_training_values = (
    train_ts["sales"]
    .to_numpy(
        dtype=float
    )
)


ets_validation_actual = (
    validation_ts["sales"]
    .to_numpy(
        dtype=float
    )
)


if len(ets_training_values) == 0:

    raise ValueError(
        "ETS training series is empty."
    )


if len(ets_validation_actual) == 0:

    raise ValueError(
        "ETS validation series is empty."
    )


# ---------------------------------------------------------
# ETS configuration
# ---------------------------------------------------------

ETS_TREND = "add"
ETS_SEASONAL = "add"
ETS_SEASONAL_PERIODS = SEASONAL_PERIODS
ETS_DAMPED_TREND = False


print(
    "\nETS configuration:"
)

print(
    "Model: ExponentialSmoothing"
)

print(
    "Trend:",
    ETS_TREND
)

print(
    "Seasonal:",
    ETS_SEASONAL
)

print(
    "Seasonal periods:",
    ETS_SEASONAL_PERIODS
)

print(
    "Damped trend:",
    ETS_DAMPED_TREND
)


# ---------------------------------------------------------
# Fit ETS model on training data only
# ---------------------------------------------------------

print(
    "\nFitting ETS model on training data..."
)


ets_model = ExponentialSmoothing(
    ets_training_values,
    trend=ETS_TREND,
    seasonal=ETS_SEASONAL,
    seasonal_periods=ETS_SEASONAL_PERIODS,
    damped_trend=ETS_DAMPED_TREND,
    initialization_method="estimated"
)


ets_fitted = ets_model.fit(
    optimized=True,
    use_brute=True
)


print(
    "ETS model fitting completed."
)


# ---------------------------------------------------------
# Generate validation forecast
# ---------------------------------------------------------

print(
    "\nGenerating ETS validation predictions..."
)


ets_validation_forecast = (
    ets_fitted
    .forecast(
        len(validation_ts)
    )
)


ets_validation_forecast = np.asarray(
    ets_validation_forecast,
    dtype=float
)


if len(ets_validation_forecast) != len(
    ets_validation_actual
):

    raise ValueError(
        "ETS validation forecast length does not "
        "match validation observations."
    )


# ---------------------------------------------------------
# Evaluate ETS validation forecast
# ---------------------------------------------------------

ets_validation_metrics = evaluate_forecast(
    ets_validation_actual,
    ets_validation_forecast
)


print(
    "\nETS validation performance:"
)

print(
    "Accuracy-style reporting is not used for "
    "this forecasting model."
)

print(
    "MAE:",
    ets_validation_metrics["MAE"]
)

print(
    "RMSE:",
    ets_validation_metrics["RMSE"]
)

print(
    "MAPE (%):",
    ets_validation_metrics["MAPE_percent"]
)

print(
    "sMAPE (%):",
    ets_validation_metrics["sMAPE_percent"]
)


# ---------------------------------------------------------
# Save ETS evaluation table
# ---------------------------------------------------------

ets_evaluation = pd.DataFrame([
    {
        "model": (
            "ETS "
            "(Additive Trend + "
            "Additive Weekly Seasonality)"
        ),
        "trend": ETS_TREND,
        "seasonal": ETS_SEASONAL,
        "seasonal_periods": ETS_SEASONAL_PERIODS,
        "damped_trend": ETS_DAMPED_TREND,
        **ets_validation_metrics
    }
])


ets_evaluation.to_csv(
    TABLES_DIR
    / "store_sales_ets_evaluation.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_ets_evaluation.csv"
)


# ---------------------------------------------------------
# Save validation forecast comparison
# ---------------------------------------------------------

ets_validation_comparison = validation_ts[
    [
        "date",
        "sales",
        "day_of_week",
        "day_of_week_name"
    ]
].copy()


ets_validation_comparison.rename(
    columns={
        "sales":
            "actual_sales"
    },
    inplace=True
)


ets_validation_comparison[
    "ets_forecast"
] = ets_validation_forecast


ets_validation_comparison[
    "ets_error"
] = (
    ets_validation_comparison[
        "actual_sales"
    ]
    -
    ets_validation_comparison[
        "ets_forecast"
    ]
)


ets_validation_comparison.to_csv(
    TABLES_DIR
    / "store_sales_ets_validation_forecast.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_ets_validation_forecast.csv"
)


# ---------------------------------------------------------
# ETS validation forecast figure
# ---------------------------------------------------------

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    validation_ts["date"],
    ets_validation_actual,
    color="black",
    linewidth=2.5,
    marker="o",
    label="Actual"
)


plt.plot(
    validation_ts["date"],
    ets_validation_forecast,
    color="darkorange",
    linewidth=2,
    marker=".",
    label="ETS Forecast"
)


plt.title(
    "Store Sales ETS Validation Forecast"
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
    / "store_sales_ets_validation_forecast.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved figure:",
    FIGURES_DIR
    / "store_sales_ets_validation_forecast.png"
)


# ---------------------------------------------------------
# ETS residual table
# ---------------------------------------------------------

ets_residuals = pd.DataFrame({
    "date":
        validation_ts["date"].to_numpy(),

    "actual_sales":
        ets_validation_actual,

    "ets_forecast":
        ets_validation_forecast,

    "residual":
        (
            ets_validation_actual
            -
            ets_validation_forecast
        ),

    "absolute_error":
        np.abs(
            ets_validation_actual
            -
            ets_validation_forecast
        ),

    "squared_error":
        (
            ets_validation_actual
            -
            ets_validation_forecast
        ) ** 2
})


ets_residuals.to_csv(
    TABLES_DIR
    / "store_sales_ets_validation_residuals.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_ets_validation_residuals.csv"
)


# =========================================================
# ETS Test-Set Forecasting
# =========================================================


# ---------------------------------------------------------
# Load Store Sales test data independently
# ---------------------------------------------------------

print(
    "\nLoading Store Sales test data..."
)


store_test = pd.read_csv(
    STORE_DIR / "test.csv",
    parse_dates=["date"]
)


print(
    "Test data shape:",
    store_test.shape
)

print(
    "Test date range:",
    store_test["date"].min(),
    "to",
    store_test["date"].max()
)


# ---------------------------------------------------------
# Create unique test dates
# ---------------------------------------------------------

test_dates = (
    store_test[
        ["date"]
    ]
    .drop_duplicates()
    .sort_values(
        "date"
    )
    .reset_index(
        drop=True
    )
)


print(
    "Test calendar observations:",
    len(test_dates)
)


# ---------------------------------------------------------
# Validate expected test period
# ---------------------------------------------------------

expected_test_dates = pd.date_range(
    TEST_START,
    TEST_END,
    freq="D"
)


actual_test_dates = pd.DatetimeIndex(
    test_dates["date"]
)


if not actual_test_dates.equals(
    expected_test_dates
):

    raise ValueError(
        "Test period does not contain the "
        "expected consecutive dates."
    )


# ---------------------------------------------------------
# Refit ETS on all available historical data
#
# Validation was used for evaluation/model selection.
# The test period is forecast only after that step.
# ---------------------------------------------------------

print(
    "\nRefitting ETS model using all historical "
    "training observations..."
)


full_training_values = (
    daily_sales_complete["sales"]
    .to_numpy(
        dtype=float
    )
)


ets_final_model = ExponentialSmoothing(
    full_training_values,
    trend=ETS_TREND,
    seasonal=ETS_SEASONAL,
    seasonal_periods=ETS_SEASONAL_PERIODS,
    damped_trend=ETS_DAMPED_TREND,
    initialization_method="estimated"
)


ets_final_fitted = ets_final_model.fit(
    optimized=True,
    use_brute=True
)


print(
    "Final ETS model fitting completed."
)


# ---------------------------------------------------------
# Generate test forecasts
# ---------------------------------------------------------

print(
    "\nGenerating ETS test forecasts..."
)


ets_test_forecast = (
    ets_final_fitted
    .forecast(
        len(test_dates)
    )
)


ets_test_forecast = np.asarray(
    ets_test_forecast,
    dtype=float
)


if len(ets_test_forecast) != len(
    test_dates
):

    raise ValueError(
        "ETS test forecast length does not "
        "match test observations."
    )


# ---------------------------------------------------------
# Create test forecast table
# ---------------------------------------------------------

ets_test_forecast_comparison = pd.DataFrame({

    "date":
        test_dates["date"],

    "day_of_week":
        test_dates[
            "date"
        ].dt.day_name(),

    "ets_forecast":
        ets_test_forecast
})


ets_test_forecast_comparison.to_csv(
    TABLES_DIR
    / "store_sales_ets_test_forecast.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_ets_test_forecast.csv"
)


# ---------------------------------------------------------
# Test forecast summary
# ---------------------------------------------------------

ets_test_forecast_summary = pd.DataFrame({

    "metric": [

        "model",

        "trend",

        "seasonal",

        "seasonal_periods",

        "damped_trend",

        "validation_start",

        "validation_end",

        "validation_observations",

        "validation_mae",

        "validation_rmse",

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

        (
            "ETS "
            "(Additive Trend + "
            "Additive Weekly Seasonality)"
        ),

        ETS_TREND,

        ETS_SEASONAL,

        ETS_SEASONAL_PERIODS,

        ETS_DAMPED_TREND,

        str(
            VALIDATION_START.date()
        ),

        str(
            VALIDATION_END.date()
        ),

        len(validation_ts),

        ets_validation_metrics[
            "MAE"
        ],

        ets_validation_metrics[
            "RMSE"
        ],

        ets_validation_metrics[
            "MAPE_percent"
        ],

        ets_validation_metrics[
            "sMAPE_percent"
        ],

        str(
            TEST_START.date()
        ),

        str(
            TEST_END.date()
        ),

        len(test_dates),

        float(
            ets_test_forecast.mean()
        ),

        float(
            ets_test_forecast.min()
        ),

        float(
            ets_test_forecast.max()
        ),

        float(
            ets_test_forecast.sum()
        )
    ]
})


ets_test_forecast_summary.to_csv(
    TABLES_DIR
    / "store_sales_ets_test_forecast_summary.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_ets_test_forecast_summary.csv"
)


# ---------------------------------------------------------
# ETS forecast figure covering test period
# ---------------------------------------------------------

plt.figure(
    figsize=(14, 7)
)


plt.plot(
    test_dates["date"],
    ets_test_forecast,
    color="darkorange",
    linewidth=2,
    marker="o",
    label="ETS Forecast"
)


plt.title(
    "Store Sales ETS Test-Period Forecast"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Forecast Total Daily Sales"
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
    / "store_sales_ets_test_forecast.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved figure:",
    FIGURES_DIR
    / "store_sales_ets_test_forecast.png"
)


# ---------------------------------------------------------
# Save model parameter information
# ---------------------------------------------------------

ets_parameter_rows = []


for parameter_name, parameter_value in (
    ets_final_fitted.params.items()
):

    if np.isscalar(parameter_value):

        ets_parameter_rows.append({
            "parameter":
                parameter_name,

            "value":
                float(parameter_value)
                if pd.notna(parameter_value)
                else np.nan
        })


ets_parameters = pd.DataFrame(
    ets_parameter_rows
)


ets_parameters.to_csv(
    TABLES_DIR
    / "store_sales_ets_parameters.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_ets_parameters.csv"
)


# ---------------------------------------------------------
# Save ETS results metadata
# ---------------------------------------------------------

ets_results = {

    "project":
        "Assignment 8",

    "script":
        "04_store_sales_ets.py",

    "dataset":
        "Store Sales Time Series Forecasting",

    "independent_execution":
        True,

    "input_training_file":
        str(
            STORE_DIR / "train.csv"
        ),

    "input_test_file":
        str(
            STORE_DIR / "test.csv"
        ),

    "calendar_validation": {

        "expected_days":
            int(len(expected_calendar)),

        "observed_days_before_completion":
            int(len(observed_dates)),

        "missing_calendar_days":
            int(len(missing_dates)),

        "missing_days_filled_with_zero":
            int(missing_sales_count)
    },

    "validation": {

        "start_date":
            str(VALIDATION_START.date()),

        "end_date":
            str(VALIDATION_END.date()),

        "observations":
            int(len(validation_ts)),

        "selection_criterion":
            MODEL_SELECTION_CRITERION
    },

    "model": {

        "type":
            "ExponentialSmoothing",

        "trend":
            ETS_TREND,

        "seasonal":
            ETS_SEASONAL,

        "seasonal_periods":
            int(ETS_SEASONAL_PERIODS),

        "damped_trend":
            bool(ETS_DAMPED_TREND)
    },

    "validation_metrics": {

        "MAE":
            ets_validation_metrics[
                "MAE"
            ],

        "RMSE":
            ets_validation_metrics[
                "RMSE"
            ],

        "MAPE_percent":
            ets_validation_metrics[
                "MAPE_percent"
            ],

        "sMAPE_percent":
            ets_validation_metrics[
                "sMAPE_percent"
            ]
    },

    "test": {

        "start_date":
            str(TEST_START.date()),

        "end_date":
            str(TEST_END.date()),

        "observations":
            int(len(test_dates))
    },

    "test_forecast_summary": {

        "mean":
            float(
                ets_test_forecast.mean()
            ),

        "minimum":
            float(
                ets_test_forecast.min()
            ),

        "maximum":
            float(
                ets_test_forecast.max()
            ),

        "total":
            float(
                ets_test_forecast.sum()
            )
    },

    "artifacts": {

        "calendar_summary":
            "store_sales_ets_calendar_summary.csv",

        "missing_calendar_dates":
            "store_sales_ets_missing_calendar_dates.csv",

        "daily_series":
            "store_sales_ets_daily_series.csv",

        "validation_summary":
            "store_sales_ets_validation_summary.csv",

        "validation_observations":
            "store_sales_ets_validation_observations.csv",

        "evaluation":
            "store_sales_ets_evaluation.csv",

        "validation_forecast":
            "store_sales_ets_validation_forecast.csv",

        "validation_residuals":
            "store_sales_ets_validation_residuals.csv",

        "test_forecast":
            "store_sales_ets_test_forecast.csv",

        "test_forecast_summary":
            "store_sales_ets_test_forecast_summary.csv",

        "parameters":
            "store_sales_ets_parameters.csv"
    }
}


with open(
    RESULTS_DIR
    / "store_sales_ets_results.json",
    "w",
    encoding="utf-8"
) as results_file:

    json.dump(
        ets_results,
        results_file,
        indent=4
    )


print(
    "Saved results:",
    RESULTS_DIR
    / "store_sales_ets_results.json"
)


# ---------------------------------------------------------
# Final artifact inventory
# ---------------------------------------------------------

print(
    "\nFinal ETS artifact inventory:"
)

print(
    "  Validation RMSE:",
    ets_validation_metrics["RMSE"]
)

print(
    "  Validation MAE:",
    ets_validation_metrics["MAE"]
)

print(
    "  Test forecast rows:",
    len(ets_test_forecast)
)

print(
    "  Missing calendar observations handled:",
    missing_sales_count
)


# ---------------------------------------------------------
# Completion
# ---------------------------------------------------------

print(
    "\n" + "=" * 60
)

print(
    "04_store_sales_ets.py completed successfully."
)

print(
    "=" * 60
)

