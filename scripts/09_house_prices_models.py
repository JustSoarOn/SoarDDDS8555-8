# ============================================================
# Assignment 8: Build & Evaluate
# Time Series, Regression, and Classification Models
#
# 09_house_prices_models.py
#
# Purpose:
#   Build and evaluate a primary regression model for the
#   House Prices: Advanced Regression Techniques dataset.
#
# Modeling objective:
#   Predict SalePrice from the available house characteristics.
#
# Primary model:
#   Random Forest Regression
#
# Validation:
#   A fixed random train/validation split is used.
#   The validation set is not used for model fitting.
#
# IMPORTANT:
#   - Test data is not used for model selection.
#   - Kaggle submission generation is intentionally excluded.
#   - Required House Prices techniques will be handled separately.
#   - This script saves durable artifacts for
#     SoarDDDS8555-8-Report.Rmd.
#   - Report.Rmd can therefore be knitted without rerunning
#     this time-consuming model.
# ============================================================


from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# Project directories
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

HOUSE_PRICES_DIR = (
    DATA_DIR
    / "house_prices_advanced_regression_techniques"
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

TARGET_COLUMN = "SalePrice"

ID_COLUMN = "Id"

VALIDATION_SIZE = 0.20

RANDOM_STATE = 8555

N_ESTIMATORS = 300

MIN_SAMPLES_LEAF = 1

MAX_FEATURES = "sqrt"

N_JOBS = -1


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def calculate_regression_metrics(
    actual,
    predicted
):
    """
    Calculate standard regression metrics.
    """

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


def calculate_log_metrics(
    actual,
    predicted
):
    """
    Calculate metrics on log1p-transformed house prices.

    These are supplemental diagnostic metrics and are not
    used as a replacement for the original-dollar metrics.
    """

    actual_log = np.log1p(
        np.maximum(
            actual,
            0
        )
    )

    predicted_log = np.log1p(
        np.maximum(
            predicted,
            0
        )
    )

    mae = mean_absolute_error(
        actual_log,
        predicted_log
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_log,
            predicted_log
        )
    )

    r2 = r2_score(
        actual_log,
        predicted_log
    )

    return {
        "log_MAE": float(mae),
        "log_RMSE": float(rmse),
        "log_R2": float(r2)
    }


# ---------------------------------------------------------
# Start
# ---------------------------------------------------------

print("=" * 60)
print("08_house_prices_models.py")
print("=" * 60)


# ---------------------------------------------------------
# Load House Prices training data
# ---------------------------------------------------------

print(
    "\nLoading House Prices training data..."
)

train_path = (
    HOUSE_PRICES_DIR
    / "train.csv"
)

if not train_path.exists():

    raise FileNotFoundError(
        "House Prices training data was not found at: "
        f"{train_path}"
    )


house_train = pd.read_csv(
    train_path
)

print(
    "Training data shape:",
    house_train.shape
)


# ---------------------------------------------------------
# Load House Prices test data
# ---------------------------------------------------------

print(
    "\nLoading House Prices test data..."
)

test_path = (
    HOUSE_PRICES_DIR
    / "test.csv"
)

if not test_path.exists():

    raise FileNotFoundError(
        "House Prices test data was not found at: "
        f"{test_path}"
    )


house_test = pd.read_csv(
    test_path
)

print(
    "Test data shape:",
    house_test.shape
)


# ---------------------------------------------------------
# Basic dataset validation
# ---------------------------------------------------------

if TARGET_COLUMN not in house_train.columns:

    raise ValueError(
        f"Required target column '{TARGET_COLUMN}' "
        "was not found in training data."
    )


if TARGET_COLUMN in house_test.columns:

    raise ValueError(
        "SalePrice unexpectedly appears in the test data."
    )


if ID_COLUMN not in house_train.columns:

    raise ValueError(
        f"Required identifier column '{ID_COLUMN}' "
        "was not found in training data."
    )


if ID_COLUMN not in house_test.columns:

    raise ValueError(
        f"Required identifier column '{ID_COLUMN}' "
        "was not found in test data."
    )


# ---------------------------------------------------------
# Save basic dataset information
# ---------------------------------------------------------

dataset_summary = pd.DataFrame({
    "dataset": [
        "training",
        "test"
    ],
    "rows": [
        len(house_train),
        len(house_test)
    ],
    "columns": [
        house_train.shape[1],
        house_test.shape[1]
    ],
    "missing_cells": [
        int(house_train.isna().sum().sum()),
        int(house_test.isna().sum().sum())
    ]
})


dataset_summary.to_csv(
    TABLES_DIR
    / "house_prices_dataset_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_dataset_summary.csv"
)


# ---------------------------------------------------------
# Target summary
# ---------------------------------------------------------

target_summary = pd.DataFrame({
    "metric": [
        "count",
        "mean",
        "median",
        "minimum",
        "maximum",
        "standard_deviation",
        "missing"
    ],
    "value": [
        int(
            house_train[TARGET_COLUMN].count()
        ),

        float(
            house_train[TARGET_COLUMN].mean()
        ),

        float(
            house_train[TARGET_COLUMN].median()
        ),

        float(
            house_train[TARGET_COLUMN].min()
        ),

        float(
            house_train[TARGET_COLUMN].max()
        ),

        float(
            house_train[TARGET_COLUMN].std()
        ),

        int(
            house_train[TARGET_COLUMN].isna().sum()
        )
    ]
})


target_summary.to_csv(
    TABLES_DIR
    / "house_prices_target_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_target_summary.csv"
)


# ---------------------------------------------------------
# Validate target
# ---------------------------------------------------------

if house_train[TARGET_COLUMN].isna().any():

    raise ValueError(
        "SalePrice contains missing values."
    )


if (
    house_train[TARGET_COLUMN] < 0
).any():

    raise ValueError(
        "SalePrice contains negative values."
    )


# ---------------------------------------------------------
# Prepare predictors and target
# ---------------------------------------------------------

X = house_train.drop(
    columns=[TARGET_COLUMN]
)

y = house_train[TARGET_COLUMN].copy()


# ---------------------------------------------------------
# Remove identifier from modeling predictors
# ---------------------------------------------------------

if ID_COLUMN in X.columns:

    X = X.drop(
        columns=[ID_COLUMN]
    )


X_test = house_test.copy()

test_ids = (
    X_test[ID_COLUMN]
    .copy()
)

X_test = X_test.drop(
    columns=[ID_COLUMN]
)


# ---------------------------------------------------------
# Identify feature types
# ---------------------------------------------------------

numeric_features = (
    X.select_dtypes(
        include=[
            "number"
        ]
    )
    .columns
    .tolist()
)

categorical_features = (
    X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    )
    .columns
    .tolist()
)


print(
    "\nFeature preparation:"
)

print(
    "Numerical feature count:",
    len(numeric_features)
)

print(
    "Categorical feature count:",
    len(categorical_features)
)


# ---------------------------------------------------------
# Save feature inventory
# ---------------------------------------------------------

feature_inventory = pd.DataFrame({
    "feature": (
        numeric_features
        + categorical_features
    ),
    "feature_type": (
        ["numeric"] * len(numeric_features)
        +
        ["categorical"] * len(categorical_features)
    )
})


feature_inventory.to_csv(
    TABLES_DIR
    / "house_prices_feature_inventory.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_feature_inventory.csv"
)


# ---------------------------------------------------------
# Train/validation split
# ---------------------------------------------------------

print(
    "\nHouse Prices validation split"
)

X_train, X_validation, y_train, y_validation = (
    train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )
)


print(
    "Training observations:",
    len(X_train)
)

print(
    "Validation observations:",
    len(X_validation)
)


# ---------------------------------------------------------
# Save validation split information
# ---------------------------------------------------------

validation_split_summary = pd.DataFrame({
    "dataset": [
        "training",
        "validation"
    ],
    "observations": [
        len(X_train),
        len(X_validation)
    ],
    "target_mean": [
        float(y_train.mean()),
        float(y_validation.mean())
    ],
    "target_median": [
        float(y_train.median()),
        float(y_validation.median())
    ]
})


validation_split_summary.to_csv(
    TABLES_DIR
    / "house_prices_validation_split_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_validation_split_summary.csv"
)


# ---------------------------------------------------------
# Numeric preprocessing
# ---------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# ---------------------------------------------------------
# Categorical preprocessing
# ---------------------------------------------------------

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


# ---------------------------------------------------------
# Combined preprocessing
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ---------------------------------------------------------
# Random Forest regression model
# ---------------------------------------------------------

random_forest = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_features=MAX_FEATURES,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)


# ---------------------------------------------------------
# Complete modeling pipeline
# ---------------------------------------------------------

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            random_forest
        )
    ]
)


# ---------------------------------------------------------
# Fit model
# ---------------------------------------------------------

print(
    "\nRandom Forest configuration:"
)

print(
    "Model: RandomForestRegressor"
)

print(
    "Number of trees:",
    N_ESTIMATORS
)

print(
    "Maximum features:",
    MAX_FEATURES
)

print(
    "Minimum samples per leaf:",
    MIN_SAMPLES_LEAF
)

print(
    "Random state:",
    RANDOM_STATE
)


print(
    "\nFitting Random Forest regression model..."
)

model_pipeline.fit(
    X_train,
    y_train
)

print(
    "Random Forest model fitting completed."
)


# ---------------------------------------------------------
# Validation predictions
# ---------------------------------------------------------

print(
    "\nGenerating House Prices validation predictions..."
)

validation_predictions = (
    model_pipeline.predict(
        X_validation
    )
)


# ---------------------------------------------------------
# Protect against negative predictions
# ---------------------------------------------------------

validation_predictions = np.maximum(
    validation_predictions,
    0
)


# ---------------------------------------------------------
# Calculate validation metrics
# ---------------------------------------------------------

validation_metrics = (
    calculate_regression_metrics(
        y_validation,
        validation_predictions
    )
)

validation_log_metrics = (
    calculate_log_metrics(
        y_validation.to_numpy(),
        validation_predictions
    )
)


print(
    "\nHouse Prices validation performance:"
)

print(
    "MAE:",
    validation_metrics["MAE"]
)

print(
    "RMSE:",
    validation_metrics["RMSE"]
)

print(
    "R-squared:",
    validation_metrics["R2"]
)

print(
    "Log MAE:",
    validation_log_metrics["log_MAE"]
)

print(
    "Log RMSE:",
    validation_log_metrics["log_RMSE"]
)

print(
    "Log R-squared:",
    validation_log_metrics["log_R2"]
)


# ---------------------------------------------------------
# Save validation evaluation
# ---------------------------------------------------------

evaluation_row = {
    "model": "Random Forest Regression",

    "validation_observations":
        int(len(y_validation)),

    "MAE":
        validation_metrics["MAE"],

    "RMSE":
        validation_metrics["RMSE"],

    "R2":
        validation_metrics["R2"],

    "log_MAE":
        validation_log_metrics["log_MAE"],

    "log_RMSE":
        validation_log_metrics["log_RMSE"],

    "log_R2":
        validation_log_metrics["log_R2"]
}


house_prices_evaluation = pd.DataFrame(
    [evaluation_row]
)


house_prices_evaluation.to_csv(
    TABLES_DIR
    / "house_prices_model_evaluation.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_model_evaluation.csv"
)


# ---------------------------------------------------------
# Create validation prediction table
# ---------------------------------------------------------

validation_predictions_table = pd.DataFrame({
    "actual_sale_price":
        y_validation.to_numpy(),

    "predicted_sale_price":
        validation_predictions
})


validation_predictions_table[
    "residual"
] = (
    validation_predictions_table[
        "actual_sale_price"
    ]
    -
    validation_predictions_table[
        "predicted_sale_price"
    ]
)


validation_predictions_table[
    "absolute_error"
] = np.abs(
    validation_predictions_table[
        "residual"
    ]
)


validation_predictions_table[
    "squared_error"
] = (
    validation_predictions_table[
        "residual"
    ]
    ** 2
)


validation_predictions_table.to_csv(
    TABLES_DIR
    / "house_prices_validation_predictions.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_validation_predictions.csv"
)


# ---------------------------------------------------------
# Validation residual summary
# ---------------------------------------------------------

residuals = (
    validation_predictions_table[
        "residual"
    ]
)


residual_summary = pd.DataFrame({
    "metric": [
        "mean_residual",
        "median_residual",
        "std_residual",
        "minimum_residual",
        "maximum_residual",
        "mean_absolute_error"
    ],
    "value": [
        float(residuals.mean()),
        float(residuals.median()),
        float(residuals.std()),
        float(residuals.min()),
        float(residuals.max()),
        float(
            np.mean(
                np.abs(residuals)
            )
        )
    ]
})


residual_summary.to_csv(
    TABLES_DIR
    / "house_prices_validation_residual_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_validation_residual_summary.csv"
)


# ---------------------------------------------------------
# Validation actual vs predicted figure
# ---------------------------------------------------------

plt.figure(
    figsize=(9, 7)
)

plt.scatter(
    validation_predictions_table[
        "actual_sale_price"
    ],
    validation_predictions_table[
        "predicted_sale_price"
    ],
    alpha=0.45,
    s=25
)


minimum_value = min(
    validation_predictions_table[
        "actual_sale_price"
    ].min(),

    validation_predictions_table[
        "predicted_sale_price"
    ].min()
)

maximum_value = max(
    validation_predictions_table[
        "actual_sale_price"
    ].max(),

    validation_predictions_table[
        "predicted_sale_price"
    ].max()
)


plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    color="red",
    linewidth=2,
    linestyle="--"
)


plt.xlabel(
    "Actual SalePrice"
)

plt.ylabel(
    "Predicted SalePrice"
)

plt.title(
    "House Prices: Actual vs. Predicted SalePrice"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "house_prices_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR
    / "house_prices_actual_vs_predicted.png"
)


# ---------------------------------------------------------
# Validation residual figure
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    validation_predictions_table[
        "predicted_sale_price"
    ],
    validation_predictions_table[
        "residual"
    ],
    alpha=0.45,
    s=25
)

plt.axhline(
    0,
    color="red",
    linestyle="--",
    linewidth=2
)

plt.xlabel(
    "Predicted SalePrice"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "House Prices: Validation Residuals"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "house_prices_validation_residuals.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR
    / "house_prices_validation_residuals.png"
)


# =========================================================
# Feature Importance, Final Model, Test Predictions,
# Results Metadata, and Artifact Inventory
# =========================================================


# ---------------------------------------------------------
# Extract transformed feature names
# ---------------------------------------------------------

print(
    "\nExtracting model feature importance..."
)

fitted_preprocessor = (
    model_pipeline.named_steps[
        "preprocessor"
    ]
)

fitted_model = (
    model_pipeline.named_steps[
        "model"
    ]
)


try:

    transformed_feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )

except Exception:

    transformed_feature_names = np.array([
        f"feature_{i + 1}"
        for i in range(
            len(
                fitted_model.feature_importances_
            )
        )
    ])


feature_importances = (
    fitted_model.feature_importances_
)


if len(transformed_feature_names) != len(
    feature_importances
):

    raise ValueError(
        "Feature-name count does not match "
        "feature-importance count."
    )


feature_importance_table = pd.DataFrame({
    "feature": transformed_feature_names,
    "importance": feature_importances
})


feature_importance_table = (
    feature_importance_table
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


feature_importance_table[
    "importance_rank"
] = np.arange(
    1,
    len(feature_importance_table) + 1
)


feature_importance_table.to_csv(
    TABLES_DIR
    / "house_prices_feature_importance.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_feature_importance.csv"
)


# ---------------------------------------------------------
# Top feature importance table
# ---------------------------------------------------------

top_feature_importances = (
    feature_importance_table
    .head(25)
    .copy()
)


top_feature_importances.to_csv(
    TABLES_DIR
    / "house_prices_top_25_feature_importance.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_top_25_feature_importance.csv"
)


# ---------------------------------------------------------
# Feature importance figure
# ---------------------------------------------------------

plot_features = (
    feature_importance_table
    .head(15)
    .sort_values(
        "importance",
        ascending=True
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    plot_features["feature"],
    plot_features["importance"],
    color="steelblue"
)

plt.xlabel(
    "Random Forest Feature Importance"
)

plt.ylabel(
    "Transformed Feature"
)

plt.title(
    "Top House Prices Model Feature Importances"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "house_prices_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR
    / "house_prices_feature_importance.png"
)


# =========================================================
# Final Model Refit
# =========================================================

print(
    "\nRefitting final Random Forest model "
    "using all available House Prices training data..."
)


final_model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                max_features=MAX_FEATURES,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS
            )
        )
    ]
)


final_model_pipeline.fit(
    X,
    y
)


print(
    "Final Random Forest model fitting completed."
)


# ---------------------------------------------------------
# Generate test predictions
# ---------------------------------------------------------

print(
    "\nGenerating House Prices test predictions..."
)


test_predictions = (
    final_model_pipeline.predict(
        X_test
    )
)


test_predictions = np.maximum(
    test_predictions,
    0
)


print(
    "Test predictions generated:",
    len(test_predictions)
)


# ---------------------------------------------------------
# Create report-ready test prediction table
# ---------------------------------------------------------

house_prices_test_predictions = pd.DataFrame({
    "Id": test_ids,
    "predicted_SalePrice": test_predictions
})


house_prices_test_predictions.to_csv(
    TABLES_DIR
    / "house_prices_test_predictions.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_test_predictions.csv"
)


# ---------------------------------------------------------
# Test prediction summary
# ---------------------------------------------------------

test_prediction_summary = pd.DataFrame({
    "metric": [
        "test_observations",
        "prediction_mean",
        "prediction_median",
        "prediction_minimum",
        "prediction_maximum",
        "prediction_standard_deviation",
        "prediction_total"
    ],
    "value": [
        int(len(test_predictions)),
        float(np.mean(test_predictions)),
        float(np.median(test_predictions)),
        float(np.min(test_predictions)),
        float(np.max(test_predictions)),
        float(np.std(test_predictions)),
        float(np.sum(test_predictions))
    ]
})


test_prediction_summary.to_csv(
    TABLES_DIR
    / "house_prices_test_prediction_summary.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_test_prediction_summary.csv"
)


# ---------------------------------------------------------
# Test prediction distribution figure
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    test_predictions,
    bins=40,
    color="steelblue",
    edgecolor="white"
)

plt.xlabel(
    "Predicted SalePrice"
)

plt.ylabel(
    "Number of Test Observations"
)

plt.title(
    "House Prices Test Prediction Distribution"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "house_prices_test_prediction_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR
    / "house_prices_test_prediction_distribution.png"
)


# =========================================================
# Save Model Configuration
# =========================================================

model_configuration = pd.DataFrame({
    "parameter": [
        "model",
        "target",
        "validation_size",
        "random_state",
        "n_estimators",
        "max_features",
        "min_samples_leaf",
        "n_jobs",
        "numeric_feature_count",
        "categorical_feature_count",
        "training_observations",
        "validation_observations",
        "test_observations"
    ],
    "value": [
        "RandomForestRegressor",
        TARGET_COLUMN,
        VALIDATION_SIZE,
        RANDOM_STATE,
        N_ESTIMATORS,
        MAX_FEATURES,
        MIN_SAMPLES_LEAF,
        N_JOBS,
        len(numeric_features),
        len(categorical_features),
        len(X_train),
        len(X_validation),
        len(X_test)
    ]
})


model_configuration.to_csv(
    TABLES_DIR
    / "house_prices_model_configuration.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_model_configuration.csv"
)


# =========================================================
# Save Results JSON
# =========================================================

house_prices_results = {

    "project":
        "Assignment 8",

    "dataset":
        "House Prices: Advanced Regression Techniques",

    "model":
        "Random Forest Regression",

    "target":
        TARGET_COLUMN,

    "validation": {

        "validation_size":
            float(VALIDATION_SIZE),

        "random_state":
            int(RANDOM_STATE),

        "training_observations":
            int(len(X_train)),

        "validation_observations":
            int(len(X_validation)),

        "MAE":
            validation_metrics["MAE"],

        "RMSE":
            validation_metrics["RMSE"],

        "R2":
            validation_metrics["R2"],

        "log_MAE":
            validation_log_metrics["log_MAE"],

        "log_RMSE":
            validation_log_metrics["log_RMSE"],

        "log_R2":
            validation_log_metrics["log_R2"]
    },

    "features": {

        "numeric_feature_count":
            int(len(numeric_features)),

        "categorical_feature_count":
            int(len(categorical_features)),

        "identifier_excluded":
            ID_COLUMN
    },

    "random_forest": {

        "n_estimators":
            int(N_ESTIMATORS),

        "max_features":
            MAX_FEATURES,

        "min_samples_leaf":
            int(MIN_SAMPLES_LEAF),

        "random_state":
            int(RANDOM_STATE)
    },

    "test": {

        "observations":
            int(len(test_predictions)),

        "prediction_mean":
            float(np.mean(test_predictions)),

        "prediction_median":
            float(np.median(test_predictions)),

        "prediction_minimum":
            float(np.min(test_predictions)),

        "prediction_maximum":
            float(np.max(test_predictions)),

        "prediction_total":
            float(np.sum(test_predictions))
    }
}


with open(
    RESULTS_DIR
    / "house_prices_model_results.json",
    "w",
    encoding="utf-8"
) as results_file:

    json.dump(
        house_prices_results,
        results_file,
        indent=4
    )


print(
    "Saved results:",
    RESULTS_DIR
    / "house_prices_model_results.json"
)


# =========================================================
# Final Artifact Inventory
# =========================================================

print(
    "\nFinal House Prices artifact inventory:"
)

print(
    "  Model:",
    "Random Forest Regression"
)

print(
    "  Validation RMSE:",
    validation_metrics["RMSE"]
)

print(
    "  Validation MAE:",
    validation_metrics["MAE"]
)

print(
    "  Validation R-squared:",
    validation_metrics["R2"]
)

print(
    "  Validation observations:",
    len(X_validation)
)

print(
    "  Test prediction rows:",
    len(test_predictions)
)

print(
    "  Feature importance rows:",
    len(feature_importance_table)
)


print(
    "\n" + "=" * 60
)

print(
    "09_house_prices_models.py completed successfully."
)

print(
    "=" * 60
)

