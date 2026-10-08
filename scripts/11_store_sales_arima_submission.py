# ============================================================
# Assignment 8: Kaggle Submission Generation
#
# 11_store_sales_arima_submission.py
#
# Purpose:
#   Generate a Kaggle-ready Store Sales submission using
#   independent ARIMA(1,1,1) forecasts for every
#   store × product-family time series.
#
# Modeling unit:
#   One ARIMA model per store × product family combination.
#
# Forecast horizon:
#   2017-08-16 through 2017-08-31
#
# Model:
#   ARIMA(1,1,1)
#
# IMPORTANT:
#   - This script is independent of 05_store_sales_arima.py.
#   - The ARIMA(1,1,1) specification was selected in 05_
#     using validation RMSE on the aggregate daily series.
#   - This script does NOT collapse the test data to 16 dates.
#   - Forecasts are generated at the store × family level.
#   - The exact IDs and row ordering from test.csv are retained.
#   - The resulting submission must contain exactly 28,512 rows.
#   - Final submission columns are exactly: id, sales.
# ============================================================


from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

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

OUTPUT_DIR = PROJECT_DIR / "output"

TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = OUTPUT_DIR / "results"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"

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

ARIMA_ORDER = (1, 1, 1)

TEST_START = pd.Timestamp(
    "2017-08-16"
)

TEST_END = pd.Timestamp(
    "2017-08-31"
)

SUBMISSION_FILENAME = (
    "store_sales_arima_submission.csv"
)


# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

def fit_arima_model(
    series,
    order
):
    """
    Fit an ARIMA model to a single store-family series.

    Warnings are suppressed because individual series may
    generate convergence or initialization warnings. Any
    actual fitting failure is handled explicitly by the
    calling code.
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
print("11_store_sales_arima_submission.py")
print("=" * 60)


# ---------------------------------------------------------
# Load training data
# ---------------------------------------------------------

print(
    "\nLoading Store Sales training data..."
)

train = pd.read_csv(
    STORE_DIR / "train.csv",
    parse_dates=["date"]
)

print(
    "Training data shape:",
    train.shape
)

print(
    "Training date range:",
    train["date"].min(),
    "to",
    train["date"].max()
)


# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------

print(
    "\nLoading Store Sales test data..."
)

test = pd.read_csv(
    STORE_DIR / "test.csv",
    parse_dates=["date"]
)

print(
    "Test data shape:",
    test.shape
)

print(
    "Test date range:",
    test["date"].min(),
    "to",
    test["date"].max()
)


# ---------------------------------------------------------
# Basic data validation
# ---------------------------------------------------------

required_train_columns = {
    "date",
    "store_nbr",
    "family",
    "sales"
}

required_test_columns = {
    "id",
    "date",
    "store_nbr",
    "family"
}

missing_train_columns = (
    required_train_columns
    - set(train.columns)
)

missing_test_columns = (
    required_test_columns
    - set(test.columns)
)

if missing_train_columns:

    raise ValueError(
        "Training data is missing required columns: "
        + str(sorted(missing_train_columns))
    )

if missing_test_columns:

    raise ValueError(
        "Test data is missing required columns: "
        + str(sorted(missing_test_columns))
    )


# ---------------------------------------------------------
# Validate test dimensions
# ---------------------------------------------------------

expected_test_rows = 28512

if len(test) != expected_test_rows:

    raise ValueError(
        f"Expected {expected_test_rows} test rows, "
        f"but found {len(test)}."
    )


# ---------------------------------------------------------
# Validate test dates
# ---------------------------------------------------------

expected_test_dates = pd.date_range(
    TEST_START,
    TEST_END,
    freq="D"
)

actual_test_dates = (
    test["date"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

expected_test_dates_series = pd.Series(
    expected_test_dates,
    name="date"
)

if not actual_test_dates.equals(
    expected_test_dates_series
):

    raise ValueError(
        "Test dates do not exactly match "
        "2017-08-16 through 2017-08-31."
    )


# ---------------------------------------------------------
# Identify stores and families
# ---------------------------------------------------------

train_stores = (
    sorted(
        train["store_nbr"]
        .dropna()
        .unique()
    )
)

train_families = (
    sorted(
        train["family"]
        .dropna()
        .unique()
    )
)

test_stores = (
    sorted(
        test["store_nbr"]
        .dropna()
        .unique()
    )
)

test_families = (
    sorted(
        test["family"]
        .dropna()
        .unique()
    )
)


print(
    "\nStore-family structure:"
)

print(
    "Unique training stores:",
    len(train_stores)
)

print(
    "Unique training families:",
    len(train_families)
)

print(
    "Unique test stores:",
    len(test_stores)
)

print(
    "Unique test families:",
    len(test_families)
)


# ---------------------------------------------------------
# Validate store/family coverage
# ---------------------------------------------------------

if set(test_stores) != set(train_stores):

    raise ValueError(
        "Test stores do not match training stores."
    )

if set(test_families) != set(train_families):

    raise ValueError(
        "Test families do not match training families."
    )


expected_combinations = (
    len(test_stores)
    * len(test_families)
)

actual_test_combinations = (
    test[
        [
            "store_nbr",
            "family"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

print(
    "Expected store-family combinations:",
    expected_combinations
)

print(
    "Test store-family combinations:",
    actual_test_combinations
)

if actual_test_combinations != expected_combinations:

    raise ValueError(
        "Test data does not contain the expected "
        "complete store-family combination structure."
    )


# ---------------------------------------------------------
# Validate test rows per store-family combination
# ---------------------------------------------------------

test_counts = (
    test
    .groupby(
        [
            "store_nbr",
            "family"
        ],
        as_index=False
    )
    .size()
)

invalid_test_counts = test_counts[
    test_counts["size"] != len(
        expected_test_dates
    )
]

if not invalid_test_counts.empty:

    raise ValueError(
        "At least one store-family combination does "
        "not contain all 16 expected test dates."
    )


# ---------------------------------------------------------
# Prepare training data
#
# Each store-family series is completed to the same
# historical daily calendar. Missing observations are filled
# with zero sales before ARIMA fitting.
# ---------------------------------------------------------

print(
    "\nPreparing store-family training series..."
)

training_calendar = pd.date_range(
    train["date"].min(),
    train["date"].max(),
    freq="D"
)

print(
    "Historical calendar days:",
    len(training_calendar)
)


# ---------------------------------------------------------
# Aggregate duplicate rows defensively
#
# The Store Sales training data normally contains one row per
# date/store/family. Grouping here makes the script robust
# against accidental duplicate records.
# ---------------------------------------------------------

train_grouped = (
    train
    .groupby(
        [
            "date",
            "store_nbr",
            "family"
        ],
        as_index=False
    )["sales"]
    .sum()
)


# ---------------------------------------------------------
# Create lookup structure for efficient series extraction
# ---------------------------------------------------------

train_grouped = (
    train_grouped
    .sort_values(
        [
            "store_nbr",
            "family",
            "date"
        ]
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# Forecast every test store-family combination
# ---------------------------------------------------------

print(
    "\nGenerating store-family ARIMA forecasts..."
)

print(
    "ARIMA order:",
    ARIMA_ORDER
)

print(
    "Forecast horizon:",
    len(expected_test_dates),
    "days"
)

print(
    "Store-family series to forecast:",
    actual_test_combinations
)

print(
    "-" * 60
)


forecast_rows = []

successful_models = 0
failed_models = 0

failure_rows = []


# ---------------------------------------------------------
# Loop through every store-family combination
# ---------------------------------------------------------

for combination_number, (
    store_nbr,
    family
) in enumerate(
    test[
        [
            "store_nbr",
            "family"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "store_nbr",
            "family"
        ]
    )
    .itertuples(
        index=False,
        name=None
    ),
    start=1
):

    combination_label = (
        f"store={store_nbr}, family={family}"
    )

    if (
        combination_number == 1
        or combination_number % 100 == 0
        or combination_number
        == actual_test_combinations
    ):

        print(
            f"Processing combination "
            f"{combination_number}/"
            f"{actual_test_combinations}: "
            f"{combination_label}"
        )


    # -----------------------------------------------------
    # Extract historical series
    # -----------------------------------------------------

    series_data = train_grouped[
        (
            train_grouped["store_nbr"]
            == store_nbr
        )
        &
        (
            train_grouped["family"]
            == family
        )
    ][
        [
            "date",
            "sales"
        ]
    ].copy()


    if series_data.empty:

        failure_rows.append({

            "store_nbr":
                store_nbr,

            "family":
                family,

            "error":
                "No historical training observations."
        })

        failed_models += 1

        continue


    # -----------------------------------------------------
    # Complete the historical calendar
    # -----------------------------------------------------

    series_data = (
        series_data
        .set_index("date")
        .reindex(training_calendar)
    )

    series_data.index.name = "date"

    series_data["sales"] = (
        series_data["sales"]
        .fillna(0.0)
    )


    historical_series = (
        series_data["sales"]
        .astype(float)
        .to_numpy()
    )


    # -----------------------------------------------------
    # Validate historical series
    # -----------------------------------------------------

    if np.isnan(
        historical_series
    ).any():

        failure_rows.append({

            "store_nbr":
                store_nbr,

            "family":
                family,

            "error":
                "Historical series contains NaN values."
        })

        failed_models += 1

        continue


    if len(historical_series) < 10:

        failure_rows.append({

            "store_nbr":
                store_nbr,

            "family":
                family,

            "error":
                "Historical series is too short."
        })

        failed_models += 1

        continue


    # -----------------------------------------------------
    # Fit ARIMA(1,1,1)
    # -----------------------------------------------------

    try:

        fitted_model = fit_arima_model(
            historical_series,
            ARIMA_ORDER
        )

        forecasts = (
            fitted_model
            .forecast(
                steps=len(
                    expected_test_dates
                )
            )
        )

        forecasts = np.asarray(
            forecasts,
            dtype=float
        )


        # -------------------------------------------------
        # Validate forecast length
        # -------------------------------------------------

        if len(forecasts) != len(
            expected_test_dates
        ):

            raise ValueError(
                "Forecast length does not match "
                "the 16-day test horizon."
            )


        # -------------------------------------------------
        # Handle numerical non-finite values
        # -------------------------------------------------

        if not np.isfinite(
            forecasts
        ).all():

            raise ValueError(
                "Forecast contains non-finite values."
            )


        # -------------------------------------------------
        # Store forecast rows
        # -------------------------------------------------

        for date_value, forecast_value in zip(
            expected_test_dates,
            forecasts
        ):

            forecast_rows.append({

                "date":
                    date_value,

                "store_nbr":
                    store_nbr,

                "family":
                    family,

                "forecast_sales":
                    float(forecast_value)
            })


        successful_models += 1


    except Exception as exc:

        failure_rows.append({

            "store_nbr":
                store_nbr,

            "family":
                family,

            "error":
                str(exc)
        })

        failed_models += 1


# ---------------------------------------------------------
# Report model-fitting results
# ---------------------------------------------------------

print(
    "\nStore-family ARIMA fitting completed."
)

print(
    "Successful models:",
    successful_models
)

print(
    "Failed models:",
    failed_models
)


# ---------------------------------------------------------
# Save fitting failures
# ---------------------------------------------------------

failure_df = pd.DataFrame(
    failure_rows,
    columns=[
        "store_nbr",
        "family",
        "error"
    ]
)

failure_df.to_csv(
    TABLES_DIR
    / "store_sales_arima_submission_model_failures.csv",
    index=False
)

print(
    "Saved model failure table:",
    TABLES_DIR
    / "store_sales_arima_submission_model_failures.csv"
)


# ---------------------------------------------------------
# Require all models to succeed
# ---------------------------------------------------------

if failed_models > 0:

    raise RuntimeError(
        f"{failed_models} store-family ARIMA models "
        "failed. Submission generation stopped so "
        "that incomplete forecasts are not submitted."
    )


# ---------------------------------------------------------
# Create forecast DataFrame
# ---------------------------------------------------------

forecast_df = pd.DataFrame(
    forecast_rows
)


# ---------------------------------------------------------
# Validate forecast row count
# ---------------------------------------------------------

expected_forecast_rows = (
    actual_test_combinations
    * len(expected_test_dates)
)

print(
    "\nForecast row validation:"
)

print(
    "Expected forecast rows:",
    expected_forecast_rows
)

print(
    "Actual forecast rows:",
    len(forecast_df)
)

if len(forecast_df) != expected_forecast_rows:

    raise ValueError(
        "Forecast row count does not match "
        "expected store-family-date combinations."
    )


# ---------------------------------------------------------
# Validate forecast uniqueness
# ---------------------------------------------------------

forecast_duplicates = (
    forecast_df
    .duplicated(
        [
            "date",
            "store_nbr",
            "family"
        ]
    )
    .sum()
)

if forecast_duplicates != 0:

    raise ValueError(
        "Forecast data contains duplicate "
        "date/store/family combinations."
    )


# ---------------------------------------------------------
# Validate forecast dates
# ---------------------------------------------------------

forecast_date_values = (
    forecast_df["date"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

if not forecast_date_values.equals(
    expected_test_dates_series
):

    raise ValueError(
        "Forecast dates do not exactly match "
        "the test period."
    )


# ---------------------------------------------------------
# Save detailed forecast artifact
# ---------------------------------------------------------

forecast_df = (
    forecast_df
    .sort_values(
        [
            "store_nbr",
            "family",
            "date"
        ]
    )
    .reset_index(drop=True)
)

forecast_df.to_csv(
    TABLES_DIR
    / "store_sales_arima_store_family_forecasts.csv",
    index=False
)

print(
    "Saved detailed forecast table:",
    TABLES_DIR
    / "store_sales_arima_store_family_forecasts.csv"
)


# ---------------------------------------------------------
# Prevent negative sales forecasts
#
# Sales cannot be negative. The ARIMA statistical forecast
# can occasionally produce a negative value, so forecasts
# are clipped at zero before submission.
#
# The unclipped forecasts remain available in the detailed
# forecast artifact above.
# ---------------------------------------------------------

negative_forecast_count = int(
    (
        forecast_df[
            "forecast_sales"
        ]
        < 0
    ).sum()
)

print(
    "\nNegative forecast validation:"
)

print(
    "Negative forecasts before clipping:",
    negative_forecast_count
)

forecast_df[
    "forecast_sales_clipped"
] = (
    forecast_df[
        "forecast_sales"
    ]
    .clip(lower=0.0)
)


# ---------------------------------------------------------
# Build lookup key for merging onto test.csv
# ---------------------------------------------------------

test_with_key = test.copy()

forecast_with_key = forecast_df.copy()


# ---------------------------------------------------------
# Merge forecasts to exact test rows
# ---------------------------------------------------------

submission_working = test_with_key.merge(
    forecast_with_key[
        [
            "date",
            "store_nbr",
            "family",
            "forecast_sales",
            "forecast_sales_clipped"
        ]
    ],
    on=[
        "date",
        "store_nbr",
        "family"
    ],
    how="left",
    validate="one_to_one"
)


# ---------------------------------------------------------
# Validate merge
# ---------------------------------------------------------

if len(submission_working) != len(test):

    raise ValueError(
        "Merged forecast data changed the number "
        "of test rows."
    )

if submission_working[
    "forecast_sales_clipped"
].isna().any():

    missing_forecasts = (
        submission_working[
            "forecast_sales_clipped"
        ]
        .isna()
        .sum()
    )

    raise ValueError(
        f"{missing_forecasts} test rows have no "
        "corresponding ARIMA forecast."
    )


# ---------------------------------------------------------
# Preserve exact test.csv row order and IDs
# ---------------------------------------------------------

submission = pd.DataFrame({

    "id":
        submission_working["id"],

    "sales":
        submission_working[
            "forecast_sales_clipped"
        ].astype(float)
})


# ---------------------------------------------------------
# Final submission validation
# ---------------------------------------------------------

print(
    "\nValidating final Kaggle submission..."
)

print(
    "-" * 60
)

print(
    "Submission rows:",
    len(submission)
)

print(
    "Submission columns:",
    list(submission.columns)
)


# Exact row count
if len(submission) != expected_test_rows:

    raise ValueError(
        f"Submission must contain exactly "
        f"{expected_test_rows} rows."
    )


# Exact columns
expected_submission_columns = [
    "id",
    "sales"
]

if list(submission.columns) != (
    expected_submission_columns
):

    raise ValueError(
        "Submission columns do not exactly match "
        "['id', 'sales']."
    )


# ID missing values
if submission["id"].isna().any():

    raise ValueError(
        "Submission contains missing IDs."
    )


# Duplicate IDs
duplicate_ids = (
    submission["id"]
    .duplicated()
    .sum()
)

if duplicate_ids != 0:

    raise ValueError(
        f"Submission contains {duplicate_ids} "
        "duplicate IDs."
    )


# Sales missing values
if submission["sales"].isna().any():

    raise ValueError(
        "Submission contains missing sales forecasts."
    )


# Sales finite values
if not np.isfinite(
    submission["sales"]
).all():

    raise ValueError(
        "Submission contains non-finite sales values."
    )


# Sales non-negative
if (
    submission["sales"] < 0
).any():

    raise ValueError(
        "Submission contains negative sales values."
    )


# IDs exactly match test.csv
test_ids = (
    test["id"]
    .reset_index(drop=True)
)

submission_ids = (
    submission["id"]
    .reset_index(drop=True)
)

if not submission_ids.equals(
    test_ids
):

    raise ValueError(
        "Submission IDs do not exactly match "
        "test.csv in row order."
    )


# ---------------------------------------------------------
# Display ID boundaries
# ---------------------------------------------------------

print(
    "First submission ID:",
    submission["id"].iloc[0]
)

print(
    "Last submission ID:",
    submission["id"].iloc[-1]
)

print(
    "Missing sales values:",
    int(
        submission["sales"].isna().sum()
    )
)

print(
    "Duplicate IDs:",
    int(
        submission["id"].duplicated().sum()
    )
)

print(
    "Minimum submitted sales:",
    float(
        submission["sales"].min()
    )
)

print(
    "Maximum submitted sales:",
    float(
        submission["sales"].max()
    )
)


# ---------------------------------------------------------
# Save Kaggle submission
# ---------------------------------------------------------

submission_path = (
    SUBMISSIONS_DIR
    / SUBMISSION_FILENAME
)

submission.to_csv(
    submission_path,
    index=False
)

print(
    "\nKaggle submission saved:"
)

print(
    submission_path
)


# ---------------------------------------------------------
# Save submission summary
# ---------------------------------------------------------

submission_summary = pd.DataFrame({

    "metric": [

        "model",

        "p",

        "d",

        "q",

        "test_start",

        "test_end",

        "test_rows",

        "store_count",

        "family_count",

        "store_family_combinations",

        "forecast_horizon_days",

        "successful_models",

        "failed_models",

        "negative_forecasts_before_clipping",

        "submission_rows",

        "submission_columns",

        "minimum_sales_forecast",

        "maximum_sales_forecast",

        "mean_sales_forecast",

        "total_sales_forecast",

        "first_id",

        "last_id",

        "submission_file"
    ],

    "value": [

        "ARIMA(1, 1, 1)",

        ARIMA_ORDER[0],

        ARIMA_ORDER[1],

        ARIMA_ORDER[2],

        str(
            TEST_START.date()
        ),

        str(
            TEST_END.date()
        ),

        len(test),

        len(test_stores),

        len(test_families),

        actual_test_combinations,

        len(expected_test_dates),

        successful_models,

        failed_models,

        negative_forecast_count,

        len(submission),

        "id,sales",

        float(
            submission["sales"].min()
        ),

        float(
            submission["sales"].max()
        ),

        float(
            submission["sales"].mean()
        ),

        float(
            submission["sales"].sum()
        ),

        int(
            submission["id"].iloc[0]
        ),

        int(
            submission["id"].iloc[-1]
        ),

        str(
            submission_path
        )
    ]
})


submission_summary.to_csv(
    TABLES_DIR
    / "store_sales_arima_submission_summary.csv",
    index=False
)

print(
    "Saved submission summary:"
)

print(
    TABLES_DIR
    / "store_sales_arima_submission_summary.csv"
)


# ---------------------------------------------------------
# Save JSON metadata
# ---------------------------------------------------------

submission_results = {

    "script":
        "11_store_sales_arima_submission.py",

    "dataset":
        "Store Sales Time Series Forecasting",

    "purpose":
        "Generate Kaggle submission",

    "model":
        "ARIMA",

    "order": {

        "p":
            ARIMA_ORDER[0],

        "d":
            ARIMA_ORDER[1],

        "q":
            ARIMA_ORDER[2]
    },

    "forecast_unit":
        "store × product family",

    "test": {

        "start_date":
            str(
                TEST_START.date()
            ),

        "end_date":
            str(
                TEST_END.date()
            ),

        "rows":
            int(
                len(test)
            ),

        "stores":
            int(
                len(test_stores)
            ),

        "families":
            int(
                len(test_families)
            ),

        "store_family_combinations":
            int(
                actual_test_combinations
            ),

        "forecast_horizon_days":
            int(
                len(expected_test_dates)
            )
    },

    "model_fitting": {

        "successful_models":
            int(
                successful_models
            ),

        "failed_models":
            int(
                failed_models
            ),

        "negative_forecasts_before_clipping":
            int(
                negative_forecast_count
            )
    },

    "submission": {

        "rows":
            int(
                len(submission)
            ),

        "columns":
            [
                "id",
                "sales"
            ],

        "first_id":
            int(
                submission["id"].iloc[0]
            ),

        "last_id":
            int(
                submission["id"].iloc[-1]
            ),

        "missing_sales":
            int(
                submission["sales"].isna().sum()
            ),

        "duplicate_ids":
            int(
                submission["id"].duplicated().sum()
            ),

        "minimum_sales":
            float(
                submission["sales"].min()
            ),

        "maximum_sales":
            float(
                submission["sales"].max()
            ),

        "mean_sales":
            float(
                submission["sales"].mean()
            ),

        "total_sales":
            float(
                submission["sales"].sum()
            ),

        "file":
            str(
                submission_path
            )
    }
}


results_path = (
    RESULTS_DIR
    / "store_sales_arima_submission_results.json"
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as results_file:

    json.dump(
        submission_results,
        results_file,
        indent=4
    )


print(
    "Saved submission metadata:"
)

print(
    results_path
)


# ---------------------------------------------------------
# Print first and last rows for visual verification
# ---------------------------------------------------------

print(
    "\nFirst 10 submission rows:"
)

print(
    submission.head(10).to_string(
        index=False
    )
)

print(
    "\nLast 10 submission rows:"
)

print(
    submission.tail(10).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Completion marker
# ---------------------------------------------------------

print(
    "\n" + "=" * 60
)

print(
    "Store Sales ARIMA Kaggle submission generation "
    "completed successfully."
)

print(
    "=" * 60
)
