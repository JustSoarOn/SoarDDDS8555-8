# ============================================================
# Assignment 8: Store Sales ETS Kaggle Submission
#
# 12_store_sales_ets_submission.py
#
# Purpose:
#   Generate a Kaggle-ready Store Sales submission using the
#   aggregate ETS forecast produced by 04_store_sales_ets.py.
#
# Kaggle submission format:
#   id,sales
#
# Expected submission:
#   28,512 rows
#
# Modeling structure:
#   - 54 stores
#   - 33 product families
#   - 1,782 store-family combinations
#   - 16 test dates
#   - 1,782 x 16 = 28,512 forecast rows
#
# ETS source artifact:
#   output/tables/store_sales_ets_test_forecast.csv
#
# Existing ETS artifact columns:
#   date
#   day_of_week
#   ets_forecast
#
# Allocation method:
#   The aggregate ETS forecast for each test date is allocated
#   across store-family combinations using each combination's
#   historical share of total aggregate sales.
#
#   Historical shares are calculated from all available training
#   observations and normalized to sum to 1.0.
#
# Important:
#   This script does NOT modify 04_store_sales_ets.py.
#   It uses the already-generated ETS forecast artifact.
# ============================================================


from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


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
# Expected test period
# ---------------------------------------------------------

TEST_START = pd.Timestamp(
    "2017-08-16"
)

TEST_END = pd.Timestamp(
    "2017-08-31"
)


# ---------------------------------------------------------
# Expected Store Sales structure
# ---------------------------------------------------------

EXPECTED_STORES = 54

EXPECTED_FAMILIES = 33

EXPECTED_COMBINATIONS = (
    EXPECTED_STORES
    * EXPECTED_FAMILIES
)

EXPECTED_TEST_ROWS = 28512

EXPECTED_TEST_DAYS = 16


# ---------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------

ETS_FORECAST_PATH = (
    TABLES_DIR
    / "store_sales_ets_test_forecast.csv"
)

SUBMISSION_PATH = (
    SUBMISSIONS_DIR
    / "store_sales_ets_submission.csv"
)

FORECAST_TABLE_PATH = (
    TABLES_DIR
    / "store_sales_ets_store_family_forecasts.csv"
)

SUMMARY_TABLE_PATH = (
    TABLES_DIR
    / "store_sales_ets_submission_summary.csv"
)

SHARE_TABLE_PATH = (
    TABLES_DIR
    / "store_sales_ets_historical_store_family_shares.csv"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "store_sales_ets_submission_results.json"
)


# =========================================================
# Start
# =========================================================

print("=" * 60)
print("12_store_sales_ets_submission.py")
print("=" * 60)


# =========================================================
# Load training data
# =========================================================

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


# =========================================================
# Load test data
# =========================================================

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


# =========================================================
# Validate training/test structure
# =========================================================

print(
    "\nStore-family structure:"
)

training_stores = sorted(
    store_train["store_nbr"]
    .unique()
)

training_families = sorted(
    store_train["family"]
    .unique()
)

test_stores = sorted(
    store_test["store_nbr"]
    .unique()
)

test_families = sorted(
    store_test["family"]
    .unique()
)

print(
    "Unique training stores:",
    len(training_stores)
)

print(
    "Unique training families:",
    len(training_families)
)

print(
    "Unique test stores:",
    len(test_stores)
)

print(
    "Unique test families:",
    len(test_families)
)


if len(training_stores) != EXPECTED_STORES:

    raise ValueError(
        f"Expected {EXPECTED_STORES} training stores, "
        f"found {len(training_stores)}."
    )


if len(training_families) != EXPECTED_FAMILIES:

    raise ValueError(
        f"Expected {EXPECTED_FAMILIES} training families, "
        f"found {len(training_families)}."
    )


if training_stores != test_stores:

    raise ValueError(
        "Training and test store sets do not match."
    )


if training_families != test_families:

    raise ValueError(
        "Training and test family sets do not match."
    )


test_combinations = (
    store_test[
        [
            "store_nbr",
            "family"
        ]
    ]
    .drop_duplicates()
)

print(
    "Expected store-family combinations:",
    EXPECTED_COMBINATIONS
)

print(
    "Test store-family combinations:",
    len(test_combinations)
)


if len(test_combinations) != EXPECTED_COMBINATIONS:

    raise ValueError(
        "Unexpected number of test store-family combinations."
    )


if len(store_test) != EXPECTED_TEST_ROWS:

    raise ValueError(
        f"Expected {EXPECTED_TEST_ROWS} test rows, "
        f"found {len(store_test)}."
    )


# =========================================================
# Validate test dates
# =========================================================

expected_test_dates = pd.date_range(
    TEST_START,
    TEST_END,
    freq="D"
)

actual_test_dates = (
    store_test["date"]
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
        "Test dates do not exactly match the expected "
        "2017-08-16 through 2017-08-31 period."
    )


# =========================================================
# Load aggregate ETS forecast
# =========================================================

print(
    "\nLoading aggregate ETS forecast artifact..."
)

if not ETS_FORECAST_PATH.exists():

    raise FileNotFoundError(
        "Required ETS forecast artifact was not found:\n"
        f"{ETS_FORECAST_PATH}\n\n"
        "Run 04_store_sales_ets.py first."
    )


ets_forecast = pd.read_csv(
    ETS_FORECAST_PATH,
    parse_dates=["date"]
)

print(
    "ETS forecast shape:",
    ets_forecast.shape
)

print(
    "ETS forecast columns:",
    list(ets_forecast.columns)
)


# ---------------------------------------------------------
# Validate actual artifact schema
#
# 04_store_sales_ets.py creates:
#   date
#   day_of_week
#   ets_forecast
# ---------------------------------------------------------

required_ets_columns = [
    "date",
    "ets_forecast"
]

missing_ets_columns = [
    column
    for column in required_ets_columns
    if column not in ets_forecast.columns
]

if missing_ets_columns:

    raise ValueError(
        "ETS forecast is missing required columns: "
        f"{missing_ets_columns}"
    )


# =========================================================
# Validate ETS forecast dates
# =========================================================

ets_forecast_dates = (
    ets_forecast["date"]
    .sort_values()
    .reset_index(drop=True)
)

if not ets_forecast_dates.equals(
    expected_test_dates_series
):

    raise ValueError(
        "ETS forecast dates do not exactly match "
        "the expected test dates."
    )


if ets_forecast["date"].duplicated().any():

    raise ValueError(
        "ETS forecast contains duplicate dates."
    )


if ets_forecast["ets_forecast"].isna().any():

    raise ValueError(
        "ETS forecast contains missing values."
    )


# ---------------------------------------------------------
# Validate ETS forecast values
# ---------------------------------------------------------

if not np.isfinite(
    ets_forecast["ets_forecast"].to_numpy(
        dtype=float
    )
).all():

    raise ValueError(
        "ETS forecast contains non-finite values."
    )


# ---------------------------------------------------------
# Sales forecasts cannot be negative for the Kaggle
# Store Sales submission.
#
# The current ETS forecast is expected to be positive.
# We nevertheless validate and clip defensively.
# ---------------------------------------------------------

negative_aggregate_forecasts = (
    ets_forecast["ets_forecast"] < 0
)

negative_count = int(
    negative_aggregate_forecasts.sum()
)

print(
    "\nAggregate ETS forecast validation:"
)

print(
    "Negative aggregate forecasts:",
    negative_count
)


if negative_count > 0:

    warnings.warn(
        "Negative aggregate ETS forecasts detected. "
        "They will be clipped to zero before allocation."
    )

    ets_forecast["ets_forecast"] = (
        ets_forecast["ets_forecast"]
        .clip(lower=0)
    )


# =========================================================
# Create complete historical store-family sales table
# =========================================================

print(
    "\nPreparing historical store-family sales shares..."
)

historical_store_family_sales = (
    store_train
    .groupby(
        [
            "store_nbr",
            "family"
        ],
        as_index=False
    )["sales"]
    .sum()
    .rename(
        columns={
            "sales":
                "historical_sales"
        }
    )
)


print(
    "Historical store-family combinations:",
    len(historical_store_family_sales)
)


if len(historical_store_family_sales) != (
    EXPECTED_COMBINATIONS
):

    raise ValueError(
        "Historical store-family combination count "
        "does not equal the expected 1,782 combinations."
    )


# =========================================================
# Calculate historical allocation shares
# =========================================================

total_historical_sales = (
    historical_store_family_sales[
        "historical_sales"
    ].sum()
)

if total_historical_sales <= 0:

    raise ValueError(
        "Total historical sales must be positive."
    )


historical_store_family_sales[
    "historical_share"
] = (
    historical_store_family_sales[
        "historical_sales"
    ]
    / total_historical_sales
)


# ---------------------------------------------------------
# Validate shares
# ---------------------------------------------------------

share_sum = (
    historical_store_family_sales[
        "historical_share"
    ].sum()
)

if not np.isclose(
    share_sum,
    1.0,
    rtol=0,
    atol=1e-10
):

    raise ValueError(
        "Historical store-family allocation shares "
        f"do not sum to 1.0. Sum={share_sum}"
    )


if (
    historical_store_family_sales[
        "historical_share"
    ] < 0
).any():

    raise ValueError(
        "Historical allocation shares cannot be negative."
    )


# ---------------------------------------------------------
# Save historical shares
# ---------------------------------------------------------

historical_store_family_sales[
    [
        "store_nbr",
        "family",
        "historical_sales",
        "historical_share"
    ]
].to_csv(
    SHARE_TABLE_PATH,
    index=False
)

print(
    "Saved historical share table:"
)

print(
    SHARE_TABLE_PATH
)


# =========================================================
# Prepare test grid
# =========================================================

print(
    "\nPreparing test forecast grid..."
)

submission_base = store_test[
    [
        "id",
        "date",
        "store_nbr",
        "family"
    ]
].copy()


# ---------------------------------------------------------
# Merge historical shares into every test row
# ---------------------------------------------------------

submission_base = submission_base.merge(
    historical_store_family_sales[
        [
            "store_nbr",
            "family",
            "historical_share"
        ]
    ],
    on=[
        "store_nbr",
        "family"
    ],
    how="left",
    validate="many_to_one"
)


if submission_base[
    "historical_share"
].isna().any():

    missing_share_rows = int(
        submission_base[
            "historical_share"
        ].isna().sum()
    )

    raise ValueError(
        "Historical allocation shares are missing for "
        f"{missing_share_rows} test rows."
    )


# =========================================================
# Merge daily aggregate ETS forecast
# =========================================================

submission_base = submission_base.merge(
    ets_forecast[
        [
            "date",
            "ets_forecast"
        ]
    ],
    on="date",
    how="left",
    validate="many_to_one"
)


if submission_base[
    "ets_forecast"
].isna().any():

    missing_forecast_rows = int(
        submission_base[
            "ets_forecast"
        ].isna().sum()
    )

    raise ValueError(
        "Aggregate ETS forecasts are missing for "
        f"{missing_forecast_rows} test rows."
    )


# =========================================================
# Generate store-family forecasts
#
# Each store-family receives:
#
#   aggregate ETS forecast
#   x
#   historical store-family share
#
# The shares sum to 1.0, so the resulting store-family
# forecasts sum back to the aggregate ETS forecast for
# every test date.
# =========================================================

print(
    "\nGenerating store-family ETS forecasts..."
)

submission_base[
    "forecast_sales"
] = (
    submission_base[
        "ets_forecast"
    ]
    *
    submission_base[
        "historical_share"
    ]
)


# ---------------------------------------------------------
# Defensive non-negative clipping
# ---------------------------------------------------------

negative_store_family_forecasts = (
    submission_base[
        "forecast_sales"
    ]
    < 0
)

negative_store_family_count = int(
    negative_store_family_forecasts.sum()
)

print(
    "Negative store-family forecasts before clipping:",
    negative_store_family_count
)


if negative_store_family_count > 0:

    submission_base[
        "forecast_sales"
    ] = (
        submission_base[
            "forecast_sales"
        ]
        .clip(lower=0)
    )


# =========================================================
# Validate forecast row count
# =========================================================

expected_forecast_rows = (
    EXPECTED_COMBINATIONS
    * EXPECTED_TEST_DAYS
)

actual_forecast_rows = (
    len(submission_base)
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
    actual_forecast_rows
)


if actual_forecast_rows != expected_forecast_rows:

    raise ValueError(
        "Forecast row count does not equal "
        f"{expected_forecast_rows}."
    )


# =========================================================
# Validate one forecast per test ID
# =========================================================

if submission_base["id"].duplicated().any():

    duplicate_id_count = int(
        submission_base["id"].duplicated().sum()
    )

    raise ValueError(
        "Duplicate test IDs found: "
        f"{duplicate_id_count}"
    )


# =========================================================
# Validate daily aggregation
#
# For every test date:
#
#   sum(store-family forecasts)
#   should equal
#   aggregate ETS forecast
#
# This is a key integrity check.
# =========================================================

daily_validation = (
    submission_base
    .groupby(
        "date",
        as_index=False
    )[
        "forecast_sales"
    ]
    .sum()
    .merge(
        ets_forecast[
            [
                "date",
                "ets_forecast"
            ]
        ],
        on="date",
        how="left",
        validate="one_to_one"
    )
)

daily_validation[
    "difference"
] = (
    daily_validation[
        "forecast_sales"
    ]
    -
    daily_validation[
        "ets_forecast"
    ]
)

daily_validation[
    "absolute_difference"
] = (
    daily_validation[
        "difference"
    ].abs()
)


max_daily_difference = float(
    daily_validation[
        "absolute_difference"
    ].max()
)

print(
    "\nDaily aggregate reconciliation:"
)

print(
    "Maximum absolute difference:",
    max_daily_difference
)


if not np.allclose(
    daily_validation[
        "forecast_sales"
    ],
    daily_validation[
        "ets_forecast"
    ],
    rtol=1e-9,
    atol=1e-6
):

    raise ValueError(
        "Store-family ETS forecasts do not reconcile "
        "to the aggregate ETS forecast."
    )


daily_validation.to_csv(
    TABLES_DIR
    / "store_sales_ets_daily_aggregate_reconciliation.csv",
    index=False
)


# =========================================================
# Create detailed forecast artifact
# =========================================================

detailed_forecast = submission_base[
    [
        "id",
        "date",
        "store_nbr",
        "family",
        "historical_share",
        "ets_forecast",
        "forecast_sales"
    ]
].copy()


detailed_forecast.to_csv(
    FORECAST_TABLE_PATH,
    index=False
)

print(
    "Saved detailed forecast table:"
)

print(
    FORECAST_TABLE_PATH
)


# =========================================================
# Construct final Kaggle submission
# =========================================================

print(
    "\nValidating final Kaggle submission..."
)

submission = submission_base[
    [
        "id",
        "forecast_sales"
    ]
].copy()

submission.rename(
    columns={
        "forecast_sales":
            "sales"
    },
    inplace=True
)


# ---------------------------------------------------------
# Validate required Kaggle columns
# ---------------------------------------------------------

expected_submission_columns = [
    "id",
    "sales"
]

if list(submission.columns) != (
    expected_submission_columns
):

    raise ValueError(
        "Submission columns do not match the required "
        "['id', 'sales'] format."
    )


# ---------------------------------------------------------
# Validate row count
# ---------------------------------------------------------

if len(submission) != EXPECTED_TEST_ROWS:

    raise ValueError(
        "Submission row count does not match the "
        f"test set. Expected {EXPECTED_TEST_ROWS}, "
        f"found {len(submission)}."
    )


# ---------------------------------------------------------
# Validate ID alignment with test.csv
# ---------------------------------------------------------

expected_ids = (
    store_test["id"]
    .reset_index(drop=True)
)

actual_ids = (
    submission["id"]
    .reset_index(drop=True)
)

if not actual_ids.equals(
    expected_ids
):

    raise ValueError(
        "Submission IDs do not exactly match the "
        "order of test.csv."
    )


# ---------------------------------------------------------
# Validate sales values
# ---------------------------------------------------------

if submission["sales"].isna().any():

    raise ValueError(
        "Submission contains missing sales values."
    )


if not np.isfinite(
    submission["sales"].to_numpy(
        dtype=float
    )
).all():

    raise ValueError(
        "Submission contains non-finite sales values."
    )


if (
    submission["sales"] < 0
).any():

    raise ValueError(
        "Submission contains negative sales values."
    )


# ---------------------------------------------------------
# Validate duplicate IDs
# ---------------------------------------------------------

duplicate_ids = (
    submission["id"].duplicated().sum()
)

if duplicate_ids != 0:

    raise ValueError(
        "Submission contains duplicate IDs: "
        f"{duplicate_ids}"
    )


# =========================================================
# Save Kaggle submission
# =========================================================

submission.to_csv(
    SUBMISSION_PATH,
    index=False
)

print(
    "\nKaggle submission saved:"
)

print(
    SUBMISSION_PATH
)


# =========================================================
# Submission summary
# =========================================================

submission_summary = pd.DataFrame({

    "metric": [

        "submission_rows",

        "submission_columns",

        "expected_rows",

        "test_dates",

        "store_count",

        "family_count",

        "store_family_combinations",

        "first_submission_id",

        "last_submission_id",

        "missing_sales_values",

        "duplicate_ids",

        "minimum_sales",

        "maximum_sales",

        "mean_sales",

        "total_submitted_sales",

        "aggregate_ets_forecast_total",

        "maximum_daily_reconciliation_difference"
    ],

    "value": [

        int(
            len(submission)
        ),

        int(
            len(submission.columns)
        ),

        EXPECTED_TEST_ROWS,

        EXPECTED_TEST_DAYS,

        EXPECTED_STORES,

        EXPECTED_FAMILIES,

        EXPECTED_COMBINATIONS,

        int(
            submission["id"].iloc[0]
        ),

        int(
            submission["id"].iloc[-1]
        ),

        int(
            submission["sales"].isna().sum()
        ),

        int(
            submission["id"].duplicated().sum()
        ),

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

        float(
            ets_forecast["ets_forecast"].sum()
        ),

        max_daily_difference
    ]
})


submission_summary.to_csv(
    SUMMARY_TABLE_PATH,
    index=False
)

print(
    "Saved submission summary:"
)

print(
    SUMMARY_TABLE_PATH
)


# =========================================================
# Save results metadata
# =========================================================

ets_results = {

    "project":
        "Assignment 8",

    "dataset":
        "Store Sales Time Series Forecasting",

    "model_family":
        "ETS",

    "source_model_script":
        "04_store_sales_ets.py",

    "source_forecast_artifact":
        str(
            ETS_FORECAST_PATH
        ),

    "allocation_method":
        "Historical store-family sales share",

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
            EXPECTED_TEST_ROWS,

        "test_days":
            EXPECTED_TEST_DAYS,

        "stores":
            EXPECTED_STORES,

        "families":
            EXPECTED_FAMILIES,

        "store_family_combinations":
            EXPECTED_COMBINATIONS
    },

    "submission": {

        "rows":
            int(
                len(submission)
            ),

        "columns":
            list(
                submission.columns
            ),

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
            )
    },

    "aggregate_ets_forecast": {

        "rows":
            int(
                len(ets_forecast)
            ),

        "mean":
            float(
                ets_forecast[
                    "ets_forecast"
                ].mean()
            ),

        "minimum":
            float(
                ets_forecast[
                    "ets_forecast"
                ].min()
            ),

        "maximum":
            float(
                ets_forecast[
                    "ets_forecast"
                ].max()
            ),

        "total":
            float(
                ets_forecast[
                    "ets_forecast"
                ].sum()
            )
    },

    "validation": {

        "daily_reconciliation":
            "passed",

        "maximum_absolute_daily_difference":
            max_daily_difference
    }
}


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as results_file:

    json.dump(
        ets_results,
        results_file,
        indent=4
    )


print(
    "Saved submission metadata:"
)

print(
    RESULTS_PATH
)


# =========================================================
# Display submission samples
# =========================================================

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


# =========================================================
# Final artifact inventory
# =========================================================

print(
    "\nFinal Store Sales ETS submission artifact inventory:"
)

print(
    "  Submission rows:",
    len(submission)
)

print(
    "  Submission columns:",
    list(submission.columns)
)

print(
    "  First submission ID:",
    submission["id"].iloc[0]
)

print(
    "  Last submission ID:",
    submission["id"].iloc[-1]
)

print(
    "  Missing sales values:",
    submission["sales"].isna().sum()
)

print(
    "  Duplicate IDs:",
    submission["id"].duplicated().sum()
)

print(
    "  Minimum submitted sales:",
    submission["sales"].min()
)

print(
    "  Maximum submitted sales:",
    submission["sales"].max()
)

print(
    "  Aggregate reconciliation:",
    "PASSED"
)


# =========================================================
# Completion marker
# =========================================================

print(
    "\n" + "=" * 60
)

print(
    "Store Sales ETS Kaggle submission generation "
    "completed successfully."
)

print(
    "=" * 60
)
