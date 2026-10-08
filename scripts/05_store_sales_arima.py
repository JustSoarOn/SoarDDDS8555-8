# ============================================================
# Assignment 8: Build & Evaluate
# Time Series, Regression, and Classification Models
#
# 05_store_sales_arima.py
#
# Purpose:
#   Build and evaluate an ARIMA-family forecasting model for
#   the Store Sales time-series dataset and save durable,
#   report-ready artifacts for SoarDDDS8555-8-Report.Rmd.
#
# Modeling unit:
#   Aggregate daily sales across ALL stores and product
#   families.
#
# Models:
#   Candidate ARIMA specifications evaluated on validation data.
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
#   - Missing calendar dates are explicitly completed with
#     zero aggregate sales before ARIMA modeling.
#   - This script calculates and saves factual results.
#   - Interpretation and report presentation are handled by
#     SoarDDDS8555-8-Report.Rmd.
# ============================================================


from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA


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


MODEL_SELECTION_CRITERION = "RMSE"


# ---------------------------------------------------------
# ARIMA candidate specifications
#
# ARIMA(p,d,q)
#
# These are deliberately modest specifications so that the
# long daily series can be fitted reproducibly without an
# unnecessarily large automated search.
# ---------------------------------------------------------

ARIMA_CANDIDATES = [
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (0, 1, 2),
    (2, 1, 1)
]


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


def fit_arima_model(
    series,
    order
):
    """
    Fit an ARIMA model to a univariate daily series.

    Warnings are suppressed during candidate-model fitting
    because some specifications may produce convergence or
    initialization warnings. Model success/failure is recorded
    explicitly in the candidate evaluation artifacts.
    """

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        model = ARIMA(
            series,
            order=order,
            trend=None
        )

        fitted_model = model.fit()

    return fitted_model


# ---------------------------------------------------------
# Start
# ---------------------------------------------------------

print("=" * 60)
print("05_store_sales_arima.py")
print("=" * 60)


# ---------------------------------------------------------
# Load training data
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
    "\nCalendar validation:"
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
    / "store_sales_arima_calendar_summary.csv",
    index=False
)

missing_calendar_dates = pd.DataFrame({
    "date": missing_dates
})

missing_calendar_dates.to_csv(
    TABLES_DIR
    / "store_sales_arima_missing_calendar_dates.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_arima_calendar_summary.csv"
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_arima_missing_calendar_dates.csv"
)


# ---------------------------------------------------------
# Complete the daily calendar
#
# Missing aggregate dates are assigned zero sales.
# This preserves a continuous daily time index required
# for ARIMA modeling.
# ---------------------------------------------------------

print(
    "\nCalendar completion:"
)

print(
    "Missing daily observations identified:",
    len(missing_dates)
)

daily_sales = (
    daily_sales
    .set_index("date")
    .reindex(calendar)
)

daily_sales.index.name = "date"

daily_sales["sales"] = (
    daily_sales["sales"]
    .fillna(0.0)
)

daily_sales = (
    daily_sales
    .reset_index()
)

print(
    "Missing daily observations filled with zero:",
    len(missing_dates)
)

print(
    "Continuous daily ARIMA series verified."
)


# ---------------------------------------------------------
# Save completed daily series
# ---------------------------------------------------------

daily_sales[
    [
        "date",
        "sales"
    ]
].to_csv(
    TABLES_DIR
    / "store_sales_arima_daily_series.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_arima_daily_series.csv"
)


# ---------------------------------------------------------
# Validate completed calendar
# ---------------------------------------------------------

if len(daily_sales) != len(calendar):

    raise ValueError(
        "Completed ARIMA daily series does not "
        "match the expected calendar length."
    )

if daily_sales["date"].duplicated().any():

    raise ValueError(
        "Completed ARIMA daily series contains "
        "duplicate dates."
    )

if daily_sales["sales"].isna().any():

    raise ValueError(
        "Completed ARIMA daily series contains "
        "missing sales values."
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

actual_validation_dates = (
    validation_ts["date"]
    .reset_index(drop=True)
)

if not actual_validation_dates.equals(
    pd.Series(
        expected_validation_dates,
        name="date"
    )
):

    raise ValueError(
        "Validation period does not contain "
        "the expected consecutive dates."
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
    / "store_sales_arima_validation_summary.csv",
    index=False
)

validation_ts[
    [
        "date",
        "sales"
    ]
].to_csv(
    TABLES_DIR
    / "store_sales_arima_validation_observations.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_arima_validation_summary.csv"
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_arima_validation_observations.csv"
)


# ---------------------------------------------------------
# Extract ARIMA training and validation arrays
# ---------------------------------------------------------

arima_training_series = (
    train_ts["sales"]
    .astype(float)
    .to_numpy()
)

actual_validation = (
    validation_ts["sales"]
    .astype(float)
    .to_numpy()
)


# ---------------------------------------------------------
# Candidate-model evaluation
# ---------------------------------------------------------

print(
    "\nARIMA candidate-model evaluation"
)

print(
    "-" * 40
)

candidate_rows = []

successful_validation_forecasts = {}


for order in ARIMA_CANDIDATES:

    order_label = (
        f"ARIMA{order}"
    )

    print(
        "\nFitting:",
        order_label
    )

    try:

        fitted_model = fit_arima_model(
            arima_training_series,
            order
        )

        validation_forecast = (
            fitted_model
            .forecast(
                steps=len(validation_ts)
            )
        )

        validation_forecast = np.asarray(
            validation_forecast,
            dtype=float
        )

        if len(validation_forecast) != len(
            actual_validation
        ):

            raise ValueError(
                "Forecast length does not match "
                "validation length."
            )

        metrics = evaluate_forecast(
            actual_validation,
            validation_forecast
        )

        candidate_rows.append({

            "model":
                order_label,

            "p":
                int(order[0]),

            "d":
                int(order[1]),

            "q":
                int(order[2]),

            "status":
                "success",

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "MAPE_percent":
                metrics["MAPE_percent"],

            "sMAPE_percent":
                metrics["sMAPE_percent"],

            "AIC":
                float(fitted_model.aic),

            "BIC":
                float(fitted_model.bic)

        })

        successful_validation_forecasts[
            order_label
        ] = validation_forecast

        print(
            "Completed:",
            order_label
        )

        print(
            "  RMSE:",
            metrics["RMSE"]
        )

        print(
            "  MAE:",
            metrics["MAE"]
        )

    except Exception as exc:

        candidate_rows.append({

            "model":
                order_label,

            "p":
                int(order[0]),

            "d":
                int(order[1]),

            "q":
                int(order[2]),

            "status":
                "failed",

            "MAE":
                np.nan,

            "RMSE":
                np.nan,

            "MAPE_percent":
                np.nan,

            "sMAPE_percent":
                np.nan,

            "AIC":
                np.nan,

            "BIC":
                np.nan,

            "error":
                str(exc)

        })

        print(
            "Candidate failed:",
            order_label
        )

        print(
            "  Error:",
            str(exc)
        )


# ---------------------------------------------------------
# Create candidate evaluation table
# ---------------------------------------------------------

arima_evaluation = pd.DataFrame(
    candidate_rows
)


# ---------------------------------------------------------
# Save candidate evaluation
# ---------------------------------------------------------

arima_evaluation.to_csv(
    TABLES_DIR
    / "store_sales_arima_model_evaluation.csv",
    index=False
)

print(
    "\nSaved table:",
    TABLES_DIR
    / "store_sales_arima_model_evaluation.csv"
)


# ---------------------------------------------------------
# Verify at least one candidate succeeded
# ---------------------------------------------------------

successful_candidates = (
    arima_evaluation[
        arima_evaluation["status"]
        == "success"
    ]
    .copy()
)

if successful_candidates.empty:

    raise RuntimeError(
        "All ARIMA candidate models failed. "
        "No validation model is available."
    )


# ---------------------------------------------------------
# Select model using validation RMSE
# ---------------------------------------------------------

successful_candidates = (
    successful_candidates
    .sort_values(
        "RMSE",
        ascending=True
    )
    .reset_index(drop=True)
)

successful_candidates[
    "validation_rank"
] = np.arange(
    1,
    len(successful_candidates) + 1
)


successful_candidates.to_csv(
    TABLES_DIR
    / "store_sales_arima_validation_ranking.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_arima_validation_ranking.csv"
)


selected_model_row = (
    successful_candidates
    .iloc[0]
)

selected_arima_model = (
    str(
        selected_model_row["model"]
    )
)

selected_arima_order = (
    int(selected_model_row["p"]),
    int(selected_model_row["d"]),
    int(selected_model_row["q"])
)

selected_validation_rmse = float(
    selected_model_row["RMSE"]
)

selected_validation_mae = float(
    selected_model_row["MAE"]
)

selected_validation_mape = float(
    selected_model_row["MAPE_percent"]
)

selected_validation_smape = float(
    selected_model_row["sMAPE_percent"]
)

print(
    "\nSelected ARIMA model:",
    selected_arima_model
)

print(
    "Model-selection criterion:",
    "Validation RMSE"
)

print(
    "Selected validation RMSE:",
    selected_validation_rmse
)

print(
    "Selected validation MAE:",
    selected_validation_mae
)


# ---------------------------------------------------------
# Retrieve selected validation forecast
# ---------------------------------------------------------

selected_validation_forecast = (
    successful_validation_forecasts[
        selected_arima_model
    ]
)


# ---------------------------------------------------------
# Create validation forecast table
# ---------------------------------------------------------

validation_forecast = validation_ts[
    [
        "date",
        "sales"
    ]
].copy()

validation_forecast.rename(
    columns={
        "sales":
            "actual_sales"
    },
    inplace=True
)

validation_forecast[
    "arima_forecast"
] = selected_validation_forecast

validation_forecast[
    "selected_model"
] = selected_arima_model


validation_forecast.to_csv(
    TABLES_DIR
    / "store_sales_arima_validation_forecast.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_arima_validation_forecast.csv"
)


# ---------------------------------------------------------
# Validation residuals
# ---------------------------------------------------------

validation_residuals = validation_forecast[
    [
        "date",
        "actual_sales",
        "arima_forecast"
    ]
].copy()

validation_residuals[
    "residual"
] = (
    validation_residuals[
        "actual_sales"
    ]
    -
    validation_residuals[
        "arima_forecast"
    ]
)

validation_residuals.to_csv(
    TABLES_DIR
    / "store_sales_arima_validation_residuals.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_arima_validation_residuals.csv"
)


# ---------------------------------------------------------
# Validation forecast figure
# ---------------------------------------------------------

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    validation_ts["date"],
    actual_validation,
    color="black",
    linewidth=2.5,
    marker="o",
    label="Actual"
)

plt.plot(
    validation_ts["date"],
    selected_validation_forecast,
    color="steelblue",
    linewidth=2,
    marker=".",
    label=selected_arima_model
)

plt.title(
    "Store Sales ARIMA Validation Forecast"
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
    / "store_sales_arima_validation_forecast.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR
    / "store_sales_arima_validation_forecast.png"
)


# ---------------------------------------------------------
# Candidate RMSE comparison figure
# ---------------------------------------------------------

plot_data = (
    successful_candidates
    .sort_values(
        "RMSE",
        ascending=True
    )
)


plt.figure(
    figsize=(11, 6)
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
    "ARIMA Model"
)

plt.title(
    "Store Sales ARIMA Validation RMSE"
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "store_sales_arima_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR
    / "store_sales_arima_rmse_comparison.png"
)


# =========================================================
# Final ARIMA Fit, Test Forecasting, Parameters,
# Results Metadata, and Artifact Inventory
# =========================================================


# ---------------------------------------------------------
# Load Store Sales test data
# ---------------------------------------------------------

print(
    "\nLoading Store Sales test data..."
)

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
# Validate expected test period
# ---------------------------------------------------------

expected_test_dates = pd.date_range(
    TEST_START,
    TEST_END,
    freq="D"
)

if not test_dates["date"].equals(
    pd.Series(
        expected_test_dates,
        name="date"
    )
):

    raise ValueError(
        "Test period does not contain the "
        "expected consecutive dates."
    )


# ---------------------------------------------------------
# Refit selected ARIMA model using all historical
# aggregate daily observations available through
# 2017-08-15.
#
# The test period remains completely unseen during
# model selection.
# ---------------------------------------------------------

print(
    "\nRefitting selected ARIMA model using "
    "all historical training observations..."
)

full_training_series = (
    daily_sales["sales"]
    .astype(float)
    .to_numpy()
)


final_arima_model = fit_arima_model(
    full_training_series,
    selected_arima_order
)

print(
    "Final ARIMA model fitting completed."
)


# ---------------------------------------------------------
# Generate test forecasts
# ---------------------------------------------------------

print(
    "\nGenerating ARIMA test forecasts..."
)

arima_test_forecasts = (
    final_arima_model
    .forecast(
        steps=len(test_dates)
    )
)

arima_test_forecasts = np.asarray(
    arima_test_forecasts,
    dtype=float
)


if len(arima_test_forecasts) != len(
    test_dates
):

    raise ValueError(
        "ARIMA test forecast length does not "
        "match the number of test dates."
    )


# ---------------------------------------------------------
# Create report-ready test forecast table
# ---------------------------------------------------------

test_forecast = pd.DataFrame({

    "date":
        test_dates["date"],

    "day_of_week":
        test_dates["date"].dt.day_name(),

    "forecast_sales":
        arima_test_forecasts,

    "selected_model":
        selected_arima_model

})


test_forecast.to_csv(
    TABLES_DIR
    / "store_sales_arima_test_forecast.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_arima_test_forecast.csv"
)


# ---------------------------------------------------------
# Test forecast summary
# ---------------------------------------------------------

test_forecast_summary = pd.DataFrame({

    "metric": [

        "selected_model",

        "p",

        "d",

        "q",

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

        selected_arima_model,

        selected_arima_order[0],

        selected_arima_order[1],

        selected_arima_order[2],

        "Validation RMSE",

        selected_validation_rmse,

        selected_validation_mae,

        selected_validation_mape,

        selected_validation_smape,

        str(
            test_dates["date"].min().date()
        ),

        str(
            test_dates["date"].max().date()
        ),

        len(test_dates),

        float(
            test_forecast[
                "forecast_sales"
            ].mean()
        ),

        float(
            test_forecast[
                "forecast_sales"
            ].min()
        ),

        float(
            test_forecast[
                "forecast_sales"
            ].max()
        ),

        float(
            test_forecast[
                "forecast_sales"
            ].sum()
        )
    ]
})


test_forecast_summary.to_csv(
    TABLES_DIR
    / "store_sales_arima_test_forecast_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_arima_test_forecast_summary.csv"
)


# ---------------------------------------------------------
# ARIMA residuals from validation period
# ---------------------------------------------------------

residual_summary = pd.DataFrame({

    "metric": [

        "mean_residual",

        "median_residual",

        "residual_std",

        "minimum_residual",

        "maximum_residual",

        "mean_absolute_residual"
    ],

    "value": [

        float(
            validation_residuals[
                "residual"
            ].mean()
        ),

        float(
            validation_residuals[
                "residual"
            ].median()
        ),

        float(
            validation_residuals[
                "residual"
            ].std()
        ),

        float(
            validation_residuals[
                "residual"
            ].min()
        ),

        float(
            validation_residuals[
                "residual"
            ].max()
        ),

        float(
            validation_residuals[
                "residual"
            ].abs().mean()
        )
    ]
})


residual_summary.to_csv(
    TABLES_DIR
    / "store_sales_arima_residual_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_arima_residual_summary.csv"
)


# ---------------------------------------------------------
# Save final ARIMA parameters
# ---------------------------------------------------------

print("\nSaving final ARIMA model parameters...")

arima_parameters = []

params = final_arima_model.params

# statsmodels may return parameters as either a pandas Series
# or a NumPy ndarray depending on the model/version.
if hasattr(params, "items"):

    for parameter_name, parameter_value in params.items():

        arima_parameters.append({
            "parameter": str(parameter_name),
            "value": float(parameter_value)
        })

else:

    # Fall back to the model's parameter names when params
    # is returned as a NumPy ndarray.
    parameter_names = getattr(
        final_arima_model,
        "param_names",
        None
    )

    if parameter_names is None:

        parameter_names = [
            f"parameter_{i + 1}"
            for i in range(len(params))
        ]

    for parameter_name, parameter_value in zip(
        parameter_names,
        params
    ):

        arima_parameters.append({
            "parameter": str(parameter_name),
            "value": float(parameter_value)
        })


arima_parameters_df = pd.DataFrame(
    arima_parameters
)


arima_parameters_df.to_csv(
    TABLES_DIR
    / "store_sales_arima_parameters.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_arima_parameters.csv"
)


# ---------------------------------------------------------
# Save model statistics
# ---------------------------------------------------------

model_statistics = pd.DataFrame({

    "metric": [

        "aic",

        "bic",

        "hqic",

        "llf",

        "nobs"
    ],

    "value": [

        float(
            final_arima_model.aic
        ),

        float(
            final_arima_model.bic
        ),

        float(
            final_arima_model.hqic
        ),

        float(
            final_arima_model.llf
        ),

        int(
            final_arima_model.nobs
        )
    ]
})


model_statistics.to_csv(
    TABLES_DIR
    / "store_sales_arima_model_statistics.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "store_sales_arima_model_statistics.csv"
)


# ---------------------------------------------------------
# Test forecast figure
# ---------------------------------------------------------

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    test_forecast["date"],
    test_forecast["forecast_sales"],
    color="steelblue",
    linewidth=2,
    marker="o",
    label=selected_arima_model
)

plt.title(
    "Store Sales ARIMA Test Forecast"
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
    / "store_sales_arima_test_forecast.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR
    / "store_sales_arima_test_forecast.png"
)


# ---------------------------------------------------------
# Save final ARIMA results metadata
# ---------------------------------------------------------

arima_results = {

    "project":
        "Assignment 8",

    "dataset":
        "Store Sales Time Series Forecasting",

    "model_family":
        "ARIMA",

    "validation": {

        "start_date":
            str(
                VALIDATION_START.date()
            ),

        "end_date":
            str(
                VALIDATION_END.date()
            ),

        "observations":
            int(
                len(validation_ts)
            ),

        "model_selection_criterion":
            "Validation RMSE"
    },

    "test": {

        "start_date":
            str(
                TEST_START.date()
            ),

        "end_date":
            str(
                TEST_END.date()
            ),

        "observations":
            int(
                len(test_dates)
            )
    },

    "calendar": {

        "expected_days":
            int(
                len(calendar)
            ),

        "observed_days_before_completion":
            int(
                len(calendar)
                - len(missing_dates)
            ),

        "missing_days":
            int(
                len(missing_dates)
            ),

        "missing_days_filled_with_zero":
            int(
                len(missing_dates)
            )
    },

    "candidate_models":
        [
            f"ARIMA{order}"
            for order
            in ARIMA_CANDIDATES
        ],

    "successful_candidate_models":
        list(
            successful_candidates[
                "model"
            ]
        ),

    "selected_model":
        selected_arima_model,

    "selected_order": {

        "p":
            selected_arima_order[0],

        "d":
            selected_arima_order[1],

        "q":
            selected_arima_order[2]
    },

    "selected_model_validation_metrics": {

        "RMSE":
            selected_validation_rmse,

        "MAE":
            selected_validation_mae,

        "MAPE_percent":
            selected_validation_mape,

        "sMAPE_percent":
            selected_validation_smape
    },

    "final_model_statistics": {

        "AIC":
            float(
                final_arima_model.aic
            ),

        "BIC":
            float(
                final_arima_model.bic
            ),

        "HQIC":
            float(
                final_arima_model.hqic
            ),

        "log_likelihood":
            float(
                final_arima_model.llf
            ),

        "observations":
            int(
                final_arima_model.nobs
            )
    },

    "primary_test_forecast": {

        "forecast_rows":
            int(
                len(test_forecast)
            ),

        "mean":
            float(
                test_forecast[
                    "forecast_sales"
                ].mean()
            ),

        "minimum":
            float(
                test_forecast[
                    "forecast_sales"
                ].min()
            ),

        "maximum":
            float(
                test_forecast[
                    "forecast_sales"
                ].max()
            ),

        "total":
            float(
                test_forecast[
                    "forecast_sales"
                ].sum()
            )
    }
}


with open(
    RESULTS_DIR
    / "store_sales_arima_results.json",
    "w",
    encoding="utf-8"
) as results_file:

    json.dump(
        arima_results,
        results_file,
        indent=4
    )


print(
    "Saved results:",
    RESULTS_DIR
    / "store_sales_arima_results.json"
)


# ---------------------------------------------------------
# Final ARIMA artifact inventory
# ---------------------------------------------------------

print(
    "\nFinal Store Sales ARIMA artifact inventory:"
)

print(
    "  Selected model:",
    selected_arima_model
)

print(
    "  Selected order:",
    selected_arima_order
)

print(
    "  Selection criterion:",
    "Validation RMSE"
)

print(
    "  Validation RMSE:",
    selected_validation_rmse
)

print(
    "  Validation MAE:",
    selected_validation_mae
)

print(
    "  Test forecast rows:",
    len(test_forecast)
)

print(
    "  Missing calendar observations handled:",
    len(missing_dates)
)


# ---------------------------------------------------------
# Completion marker
# ---------------------------------------------------------

print(
    "\n" + "=" * 60
)

print(
    "\n05_store_sales_arima.py completed successfully."
)

print(
    "=" * 60
)

