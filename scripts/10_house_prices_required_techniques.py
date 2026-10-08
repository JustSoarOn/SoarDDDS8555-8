# ============================================================
# Assignment 8: Build & Evaluate
# Time Series, Regression, and Classification Models
#
# 10_house_prices_required_techniques.py
#
# Purpose:
#   Address the specifically required House Prices regression
#   techniques in Assignment 8.
#
# Required techniques addressed:
#   1. Dichotomous variables
#   2. Interaction terms
#   3. Polynomial terms
#   4. Subset investigation / feature selection
#   5. Response transformation
#   6. Dimension reduction
#   7. Two regression models
#   8. Assumption investigation
#
# Models:
#
#   Model A:
#       Engineered Linear Regression
#       - log-transformed SalePrice
#       - dichotomous variables / one-hot encoding
#       - polynomial terms
#       - interaction terms
#       - feature subset selection
#
#   Model B:
#       PCA Ridge Regression
#       - log-transformed SalePrice
#       - one-hot encoded predictors
#       - standardized predictors
#       - PCA dimension reduction
#       - Ridge regression
#
# IMPORTANT:
#   - The Kaggle test set is never used for model selection.
#   - The validation set is used only for model comparison.
#   - Test predictions are generated only after the models have
#     been evaluated.
#   - This script creates durable artifacts for Report.Rmd.
#   - Kaggle submission files are created for the final models.
# ============================================================


from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler
)


warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)


# ---------------------------------------------------------
# Project directories
# ---------------------------------------------------------

PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
)

DATA_DIR = (
    PROJECT_DIR
    / "data"
)

HOUSE_PRICES_DIR = (
    DATA_DIR
    / "house_prices_advanced_regression_techniques"
)


# ---------------------------------------------------------
# Output directories
# ---------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
)

FIGURES_DIR = (
    OUTPUT_DIR
    / "figures"
)

TABLES_DIR = (
    OUTPUT_DIR
    / "tables"
)

RESULTS_DIR = (
    OUTPUT_DIR
    / "results"
)

SUBMISSIONS_DIR = (
    OUTPUT_DIR
    / "submissions"
)


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

TARGET_COLUMN = (
    "SalePrice"
)

ID_COLUMN = (
    "Id"
)

VALIDATION_SIZE = (
    0.20
)

RANDOM_STATE = (
    8555
)


# ---------------------------------------------------------
# Feature-selection settings
# ---------------------------------------------------------

# Number of encoded predictors retained by the subset
# investigation for Model A.
SELECT_K_BEST = (
    40
)


# ---------------------------------------------------------
# PCA settings
# ---------------------------------------------------------

# PCA retains enough components to explain approximately
# this proportion of predictor variance.
PCA_VARIANCE = (
    0.95
)


# ---------------------------------------------------------
# Ridge settings
# ---------------------------------------------------------

RIDGE_ALPHA = (
    10.0
)


# ---------------------------------------------------------
# Engineered numeric variables
# ---------------------------------------------------------

# These variables are deliberately selected from the
# House Prices data because they represent interpretable
# measures of size, quality, age, and garage capacity.

ENGINEERED_NUMERIC_COLUMNS = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
    "YearRemodAdd"
]


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def regression_metrics(
    actual,
    predicted
):
    """
    Calculate regression metrics on the original scale.
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


def log_scale_metrics(
    actual,
    predicted
):
    """
    Calculate metrics after log1p transformation.
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


def create_engineered_features(
    data
):
    """
    Create explicit polynomial and interaction features.

    These features are deliberately created before the
    general preprocessing pipeline so that the required
    polynomial and interaction techniques are transparent
    and reportable.
    """

    data = data.copy()

    required_columns = [
        "OverallQual",
        "GrLivArea",
        "TotalBsmtSF",
        "GarageCars",
        "YearBuilt",
        "YearRemodAdd"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required engineering columns are missing: "
            + ", ".join(missing_columns)
        )

    # -----------------------------------------------------
    # Polynomial terms
    # -----------------------------------------------------

    data[
        "OverallQual_squared"
    ] = (
        data["OverallQual"]
        ** 2
    )

    data[
        "GrLivArea_squared"
    ] = (
        data["GrLivArea"]
        ** 2
    )

    data[
        "TotalBsmtSF_squared"
    ] = (
        data["TotalBsmtSF"]
        ** 2
    )

    # -----------------------------------------------------
    # Interaction terms
    # -----------------------------------------------------

    data[
        "OverallQual_x_GrLivArea"
    ] = (
        data["OverallQual"]
        *
        data["GrLivArea"]
    )

    data[
        "OverallQual_x_TotalBsmtSF"
    ] = (
        data["OverallQual"]
        *
        data["TotalBsmtSF"]
    )

    data[
        "GrLivArea_x_GarageCars"
    ] = (
        data["GrLivArea"]
        *
        data["GarageCars"]
    )

    # -----------------------------------------------------
    # Age variables
    # -----------------------------------------------------

    data[
        "HouseAgeAtSale"
    ] = (
        data["YrSold"]
        -
        data["YearBuilt"]
    )

    data[
        "YearsSinceRemodel"
    ] = (
        data["YrSold"]
        -
        data["YearRemodAdd"]
    )

    data[
        "HouseAgeAtSale"
    ] = data[
        "HouseAgeAtSale"
    ].clip(
        lower=0
    )

    data[
        "YearsSinceRemodel"
    ] = data[
        "YearsSinceRemodel"
    ].clip(
        lower=0
    )

    return data


def make_one_hot_encoder():
    """
    Create a version-compatible OneHotEncoder.
    """

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True
        )

def make_dense_one_hot_encoder():
    """
    Create a dense OneHotEncoder for PCA.

    PCA with svd_solver="full" requires dense input,
    so Model B uses dense one-hot encoded predictors.
    """

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )




# ---------------------------------------------------------
# Start
# ---------------------------------------------------------

print(
    "=" * 60
)

print(
    "10_house_prices_required_techniques.py"
)

print(
    "=" * 60
)


# ---------------------------------------------------------
# Load training data
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
# Load test data
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
# Basic validation
# ---------------------------------------------------------

if TARGET_COLUMN not in house_train.columns:

    raise ValueError(
        "SalePrice was not found in training data."
    )


if TARGET_COLUMN in house_test.columns:

    raise ValueError(
        "SalePrice unexpectedly appears in test data."
    )


if ID_COLUMN not in house_train.columns:

    raise ValueError(
        "Id was not found in training data."
    )


if ID_COLUMN not in house_test.columns:

    raise ValueError(
        "Id was not found in test data."
    )


if house_train[
    TARGET_COLUMN
].isna().any():

    raise ValueError(
        "SalePrice contains missing observations."
    )


if (
    house_train[
        TARGET_COLUMN
    ]
    < 0
).any():

    raise ValueError(
        "SalePrice contains negative values."
    )


# ---------------------------------------------------------
# Target
# ---------------------------------------------------------

y = (
    house_train[
        TARGET_COLUMN
    ]
    .copy()
)


# ---------------------------------------------------------
# Preserve test identifiers
# ---------------------------------------------------------

test_ids = (
    house_test[
        ID_COLUMN
    ]
    .copy()
)


# ---------------------------------------------------------
# Create modeling data
# ---------------------------------------------------------

X_raw = (
    house_train
    .drop(
        columns=[
            TARGET_COLUMN,
            ID_COLUMN
        ]
    )
    .copy()
)


X_test_raw = (
    house_test
    .drop(
        columns=[
            ID_COLUMN
        ]
    )
    .copy()
)


# ---------------------------------------------------------
# Explicitly create required engineered terms
# ---------------------------------------------------------

print(
    "\nCreating required polynomial and interaction terms..."
)


X_engineered = (
    create_engineered_features(
        X_raw
    )
)


X_test_engineered = (
    create_engineered_features(
        X_test_raw
    )
)


print(
    "Polynomial terms created:"
)

print(
    "  OverallQual_squared"
)

print(
    "  GrLivArea_squared"
)

print(
    "  TotalBsmtSF_squared"
)


print(
    "Interaction terms created:"
)

print(
    "  OverallQual_x_GrLivArea"
)

print(
    "  OverallQual_x_TotalBsmtSF"
)

print(
    "  GrLivArea_x_GarageCars"
)


print(
    "Additional derived age variables created:"
)

print(
    "  HouseAgeAtSale"
)

print(
    "  YearsSinceRemodel"
)


# ---------------------------------------------------------
# Save engineered feature inventory
# ---------------------------------------------------------

engineered_feature_names = [
    "OverallQual_squared",
    "GrLivArea_squared",
    "TotalBsmtSF_squared",
    "OverallQual_x_GrLivArea",
    "OverallQual_x_TotalBsmtSF",
    "GrLivArea_x_GarageCars",
    "HouseAgeAtSale",
    "YearsSinceRemodel"
]


engineered_inventory = pd.DataFrame({
    "feature": engineered_feature_names,
    "technique": [
        "Polynomial",
        "Polynomial",
        "Polynomial",
        "Interaction",
        "Interaction",
        "Interaction",
        "Derived variable",
        "Derived variable"
    ]
})


engineered_inventory.to_csv(
    TABLES_DIR
    / "house_prices_required_engineered_features.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_engineered_features.csv"
)


# ---------------------------------------------------------
# Create log-transformed response
# ---------------------------------------------------------

print(
    "\nCreating log-transformed SalePrice response..."
)


y_log = np.log1p(
    y
)


log_target_summary = pd.DataFrame({
    "metric": [
        "original_mean",
        "original_median",
        "original_std",
        "log_mean",
        "log_median",
        "log_std"
    ],
    "value": [
        float(y.mean()),
        float(y.median()),
        float(y.std()),
        float(y_log.mean()),
        float(y_log.median()),
        float(y_log.std())
    ]
})


log_target_summary.to_csv(
    TABLES_DIR
    / "house_prices_required_target_transformation.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_target_transformation.csv"
)


# ---------------------------------------------------------
# Train/validation split
# ---------------------------------------------------------

print(
    "\nHouse Prices required-techniques validation split"
)


(
    X_train,
    X_validation,
    y_train,
    y_validation,
    y_log_train,
    y_log_validation
) = train_test_split(
    X_engineered,
    y,
    y_log,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE
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
# Save split information
# ---------------------------------------------------------

validation_summary = pd.DataFrame({
    "dataset": [
        "training",
        "validation"
    ],
    "observations": [
        len(X_train),
        len(X_validation)
    ],
    "saleprice_mean": [
        float(y_train.mean()),
        float(y_validation.mean())
    ],
    "saleprice_median": [
        float(y_train.median()),
        float(y_validation.median())
    ],
    "log_saleprice_mean": [
        float(y_log_train.mean()),
        float(y_log_validation.mean())
    ],
    "log_saleprice_median": [
        float(y_log_train.median()),
        float(y_log_validation.median())
    ]
})


validation_summary.to_csv(
    TABLES_DIR
    / "house_prices_required_validation_summary.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_validation_summary.csv"
)


# ---------------------------------------------------------
# Identify numeric and categorical predictors
# ---------------------------------------------------------

numeric_features = (
    X_train
    .select_dtypes(
        include=[
            "number"
        ]
    )
    .columns
    .tolist()
)


categorical_features = (
    X_train
    .select_dtypes(
        include=[
            "object",
            "category",
            "bool",
            "str"
        ]
    )
    .columns
    .tolist()
)


print(
    "\nRequired-techniques feature inventory:"
)


print(
    "Numerical predictors:",
    len(numeric_features)
)


print(
    "Categorical predictors:",
    len(categorical_features)
)


print(
    "Dichotomous variables will be represented "
    "through one-hot encoding."
)


# ---------------------------------------------------------
# Save feature inventory
# ---------------------------------------------------------

required_feature_inventory = pd.DataFrame({
    "feature": (
        numeric_features
        +
        categorical_features
    ),
    "feature_type": (
        ["numeric"] * len(numeric_features)
        +
        ["categorical"] * len(categorical_features)
    )
})


required_feature_inventory.to_csv(
    TABLES_DIR
    / "house_prices_required_feature_inventory.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_feature_inventory.csv"
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
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ---------------------------------------------------------
# Categorical preprocessing - sparse
#
# Used by Model A.
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
            make_one_hot_encoder()
        )
    ]
)


# ---------------------------------------------------------
# General sparse encoded predictor transformer
#
# Used by Model A.
# ---------------------------------------------------------

encoded_preprocessor = ColumnTransformer(
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
# Dense categorical preprocessing
#
# Used by Model B because PCA with the full SVD solver
# requires dense input.
# ---------------------------------------------------------

dense_categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            make_dense_one_hot_encoder()
        )
    ]
)


# ---------------------------------------------------------
# Dense encoded predictor transformer
#
# Used by Model B.
# ---------------------------------------------------------

dense_encoded_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),

        (
            "categorical",
            dense_categorical_transformer,
            categorical_features
        )
    ]
)


# ---------------------------------------------------------
# Model A preprocessing
#
# Model A deliberately includes:
#   - numeric predictors
#   - dichotomous variables
#   - polynomial terms
#   - interactions
#   - feature subset selection
# ---------------------------------------------------------

model_a_preprocessor = (
    encoded_preprocessor
)


model_a = Pipeline(
    steps=[
        (
            "preprocessor",
            model_a_preprocessor
        ),

        (
            "subset_selection",
            SelectKBest(
                score_func=f_regression,
                k=SELECT_K_BEST
            )
        ),

        (
            "regression",
            LinearRegression()
        )
    ]
)


# ---------------------------------------------------------
# Model B preprocessing
#
# Model B deliberately includes:
#   - dichotomous variables
#   - standardized predictors
#   - PCA dimension reduction
#   - Ridge regression
#
# PCA uses dense one-hot encoded data because the
# full SVD solver does not accept sparse matrices.
# ---------------------------------------------------------

model_b = Pipeline(
    steps=[
        (
            "preprocessor",
            dense_encoded_preprocessor
        ),

        (
            "pca",
            PCA(
                n_components=PCA_VARIANCE,
                svd_solver="full",
                random_state=RANDOM_STATE
            )
        ),

        (
            "regression",
            Ridge(
                alpha=RIDGE_ALPHA
            )
        )
    ]
)





# ---------------------------------------------------------
# Model configuration summary
# ---------------------------------------------------------

model_configuration = pd.DataFrame({
    "model": [
        "Model A - Engineered Linear Regression",
        "Model B - PCA Ridge Regression"
    ],

    "response": [
        "log1p(SalePrice)",
        "log1p(SalePrice)"
    ],

    "dichotomous_variables": [
        "One-hot encoding",
        "One-hot encoding"
    ],

    "polynomial_terms": [
        "Yes",
        "No"
    ],

    "interaction_terms": [
        "Yes",
        "No"
    ],

    "subset_selection": [
        f"SelectKBest, k={SELECT_K_BEST}",
        "No"
    ],

    "dimension_reduction": [
        "No",
        f"PCA, variance={PCA_VARIANCE}"
    ],

    "regularization": [
        "None",
        f"Ridge alpha={RIDGE_ALPHA}"
    ]
})


model_configuration.to_csv(
    TABLES_DIR
    / "house_prices_required_model_configuration.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_model_configuration.csv"
)


# ---------------------------------------------------------
# Fit Model A
# ---------------------------------------------------------

print(
    "\n============================================================"
)

print(
    "MODEL A - ENGINEERED LINEAR REGRESSION"
)

print(
    "============================================================"
)


print(
    "\nFitting Model A using log1p(SalePrice)..."
)


model_a.fit(
    X_train,
    y_log_train
)


print(
    "Model A fitting completed."
)


# ---------------------------------------------------------
# Model A validation predictions
# ---------------------------------------------------------

model_a_log_predictions = (
    model_a.predict(
        X_validation
    )
)


model_a_predictions = (
    np.expm1(
        model_a_log_predictions
    )
)


model_a_predictions = np.maximum(
    model_a_predictions,
    0
)


# ---------------------------------------------------------
# Model A metrics
# ---------------------------------------------------------

model_a_original_metrics = (
    regression_metrics(
        y_validation,
        model_a_predictions
    )
)


model_a_log_metrics = (
    log_scale_metrics(
        y_validation.to_numpy(),
        model_a_predictions
    )
)


print(
    "\nModel A validation performance:"
)


print(
    "MAE:",
    model_a_original_metrics["MAE"]
)


print(
    "RMSE:",
    model_a_original_metrics["RMSE"]
)


print(
    "R-squared:",
    model_a_original_metrics["R2"]
)


print(
    "Log MAE:",
    model_a_log_metrics["log_MAE"]
)


print(
    "Log RMSE:",
    model_a_log_metrics["log_RMSE"]
)


print(
    "Log R-squared:",
    model_a_log_metrics["log_R2"]
)


# ---------------------------------------------------------
# Model A selected subset
# ---------------------------------------------------------

model_a_selector = (
    model_a.named_steps[
        "subset_selection"
    ]
)


model_a_encoded_names = (
    model_a.named_steps[
        "preprocessor"
    ]
    .get_feature_names_out()
)


model_a_support = (
    model_a_selector
    .get_support()
)


model_a_selected_features = (
    model_a_encoded_names[
        model_a_support
    ]
)


# ---------------------------------------------------------
# Verify required engineered terms survived feature selection
# ---------------------------------------------------------

required_engineered_terms = [
    "OverallQual_squared",
    "GrLivArea_squared",
    "TotalBsmtSF_squared",
    "OverallQual_x_GrLivArea",
    "OverallQual_x_TotalBsmtSF",
    "GrLivArea_x_GarageCars"
]


engineered_selection_check = []

for term in required_engineered_terms:

    matches = [
        feature
        for feature in model_a_selected_features
        if term in feature
    ]

    engineered_selection_check.append({
        "engineered_term": term,
        "selected": len(matches) > 0,
        "matching_selected_features": ", ".join(matches)
    })


engineered_selection_check = pd.DataFrame(
    engineered_selection_check
)


engineered_selection_check.to_csv(
    TABLES_DIR
    / "house_prices_required_engineered_term_selection.csv",
    index=False
)


print(
    "\nRequired engineered-term selection check:"
)

print(
    engineered_selection_check.to_string(
        index=False
    )
)


model_a_feature_scores = (
    model_a_selector
    .scores_
)


model_a_subset_table = pd.DataFrame({
    "feature": model_a_encoded_names,
    "f_score": model_a_feature_scores,
    "selected": model_a_support
})


model_a_subset_table = (
    model_a_subset_table
    .sort_values(
        "f_score",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


model_a_subset_table[
    "selection_rank"
] = np.arange(
    1,
    len(model_a_subset_table) + 1
)


model_a_subset_table.to_csv(
    TABLES_DIR
    / "house_prices_required_model_a_subset_selection.csv",
    index=False
)


print(
    "\nModel A selected predictor count:",
    len(model_a_selected_features)
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_model_a_subset_selection.csv"
)


# ---------------------------------------------------------
# Model A coefficient table
# ---------------------------------------------------------

model_a_coefficients = (
    model_a.named_steps[
        "regression"
    ].coef_
)


if len(model_a_coefficients) != len(
    model_a_selected_features
):

    raise ValueError(
        "Model A coefficient count does not match "
        "selected feature count."
    )


model_a_coefficient_table = pd.DataFrame({
    "feature": model_a_selected_features,
    "coefficient": model_a_coefficients
})


model_a_coefficient_table[
    "absolute_coefficient"
] = np.abs(
    model_a_coefficient_table[
        "coefficient"
    ]
)


model_a_coefficient_table = (
    model_a_coefficient_table
    .sort_values(
        "absolute_coefficient",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


model_a_coefficient_table.to_csv(
    TABLES_DIR
    / "house_prices_required_model_a_coefficients.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_model_a_coefficients.csv"
)


# =========================================================
# MODEL B - PCA RIDGE REGRESSION
# =========================================================

print(
    "\n============================================================"
)

print(
    "MODEL B - PCA RIDGE REGRESSION"
)

print(
    "============================================================"
)


print(
    "\nFitting Model B using standardized predictors, "
    "PCA dimension reduction, and Ridge regression..."
)


model_b.fit(
    X_train,
    y_log_train
)


print(
    "Model B fitting completed."
)


# ---------------------------------------------------------
# Model B validation predictions
# ---------------------------------------------------------

model_b_log_predictions = (
    model_b.predict(
        X_validation
    )
)


model_b_predictions = (
    np.expm1(
        model_b_log_predictions
    )
)


model_b_predictions = np.maximum(
    model_b_predictions,
    0
)


# ---------------------------------------------------------
# Model B validation metrics
# ---------------------------------------------------------

model_b_original_metrics = (
    regression_metrics(
        y_validation,
        model_b_predictions
    )
)


model_b_log_metrics = (
    log_scale_metrics(
        y_validation.to_numpy(),
        model_b_predictions
    )
)


print(
    "\nModel B validation performance:"
)


print(
    "MAE:",
    model_b_original_metrics["MAE"]
)


print(
    "RMSE:",
    model_b_original_metrics["RMSE"]
)


print(
    "R-squared:",
    model_b_original_metrics["R2"]
)


print(
    "Log MAE:",
    model_b_log_metrics["log_MAE"]
)


print(
    "Log RMSE:",
    model_b_log_metrics["log_RMSE"]
)


print(
    "Log R-squared:",
    model_b_log_metrics["log_R2"]
)


# ---------------------------------------------------------
# PCA diagnostics
# ---------------------------------------------------------

fitted_pca = (
    model_b.named_steps[
        "pca"
    ]
)


explained_variance_ratio = (
    fitted_pca
    .explained_variance_ratio_
)


cumulative_explained_variance = (
    np.cumsum(
        explained_variance_ratio
    )
)


pca_component_table = pd.DataFrame({
    "component": np.arange(
        1,
        len(
            explained_variance_ratio
        ) + 1
    ),

    "explained_variance_ratio":
        explained_variance_ratio,

    "cumulative_explained_variance":
        cumulative_explained_variance
})


pca_component_table.to_csv(
    TABLES_DIR
    / "house_prices_required_pca_components.csv",
    index=False
)


print(
    "\nPCA dimension-reduction results:"
)


print(
    "Original encoded predictor count:",
    len(
        model_b.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    )
)


print(
    "Retained PCA components:",
    fitted_pca.n_components_
)


print(
    "Cumulative explained variance:",
    cumulative_explained_variance[-1]
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_pca_components.csv"
)


# ---------------------------------------------------------
# PCA scree / cumulative variance figure
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    pca_component_table[
        "component"
    ],
    pca_component_table[
        "cumulative_explained_variance"
    ],
    color="steelblue",
    linewidth=2
)


plt.axhline(
    PCA_VARIANCE,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Target variance = {PCA_VARIANCE:.0%}"
)


plt.xlabel(
    "Number of Principal Components"
)


plt.ylabel(
    "Cumulative Explained Variance"
)


plt.title(
    "House Prices PCA Cumulative Explained Variance"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    FIGURES_DIR
    / "house_prices_required_pca_variance.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved figure:",
    FIGURES_DIR
    / "house_prices_required_pca_variance.png"
)


# =========================================================
# MODEL COMPARISON
# =========================================================

print(
    "\n============================================================"
)

print(
    "HOUSE PRICES REQUIRED MODEL COMPARISON"
)

print(
    "============================================================"
)


model_comparison = pd.DataFrame([
    {
        "model":
            "Model A - Engineered Linear Regression",

        "MAE":
            model_a_original_metrics["MAE"],

        "RMSE":
            model_a_original_metrics["RMSE"],

        "R2":
            model_a_original_metrics["R2"],

        "log_MAE":
            model_a_log_metrics["log_MAE"],

        "log_RMSE":
            model_a_log_metrics["log_RMSE"],

        "log_R2":
            model_a_log_metrics["log_R2"]
    },

    {
        "model":
            "Model B - PCA Ridge Regression",

        "MAE":
            model_b_original_metrics["MAE"],

        "RMSE":
            model_b_original_metrics["RMSE"],

        "R2":
            model_b_original_metrics["R2"],

        "log_MAE":
            model_b_log_metrics["log_MAE"],

        "log_RMSE":
            model_b_log_metrics["log_RMSE"],

        "log_R2":
            model_b_log_metrics["log_R2"]
    }
])


model_comparison.to_csv(
    TABLES_DIR
    / "house_prices_required_model_comparison.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_model_comparison.csv"
)


# ---------------------------------------------------------
# Determine validation model based on log RMSE
# ---------------------------------------------------------

if (
    model_a_log_metrics["log_RMSE"]
    <=
    model_b_log_metrics["log_RMSE"]
):

    selected_model_name = (
        "Model A - Engineered Linear Regression"
    )

    selected_model_label = (
        "model_a"
    )

    selected_validation_log_rmse = (
        model_a_log_metrics["log_RMSE"]
    )

    selected_validation_rmse = (
        model_a_original_metrics["RMSE"]
    )

    selected_validation_mae = (
        model_a_original_metrics["MAE"]
    )

    selected_validation_r2 = (
        model_a_original_metrics["R2"]
    )

else:

    selected_model_name = (
        "Model B - PCA Ridge Regression"
    )

    selected_model_label = (
        "model_b"
    )

    selected_validation_log_rmse = (
        model_b_log_metrics["log_RMSE"]
    )

    selected_validation_rmse = (
        model_b_original_metrics["RMSE"]
    )

    selected_validation_mae = (
        model_b_original_metrics["MAE"]
    )

    selected_validation_r2 = (
        model_b_original_metrics["R2"]
    )


print(
    "\nValidation comparison criterion:"
)

print(
    "Log RMSE"
)


print(
    "Selected model for final refit:",
    selected_model_name
)


print(
    "Selected validation Log RMSE:",
    selected_validation_log_rmse
)


# ---------------------------------------------------------
# Save explicit comparison summary
# ---------------------------------------------------------

model_selection_summary = pd.DataFrame({
    "criterion": [
        "Validation Log RMSE"
    ],

    "selected_model": [
        selected_model_name
    ],

    "selected_model_internal_label": [
        selected_model_label
    ],

    "selected_log_RMSE": [
        selected_validation_log_rmse
    ],

    "selected_original_RMSE": [
        selected_validation_rmse
    ],

    "selected_original_MAE": [
        selected_validation_mae
    ],

    "selected_original_R2": [
        selected_validation_r2
    ]
})


model_selection_summary.to_csv(
    TABLES_DIR
    / "house_prices_required_model_selection.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_model_selection.csv"
)


# =========================================================
# Validation Residual Diagnostics
# =========================================================

print(
    "\nGenerating residual diagnostics..."
)


validation_results = pd.DataFrame({
    "actual_sale_price":
        y_validation.to_numpy(),

    "model_a_prediction":
        model_a_predictions,

    "model_b_prediction":
        model_b_predictions,

    "model_a_log_prediction":
        model_a_log_predictions,

    "model_b_log_prediction":
        model_b_log_predictions
})


validation_results[
    "model_a_residual"
] = (
    validation_results[
        "actual_sale_price"
    ]
    -
    validation_results[
        "model_a_prediction"
    ]
)


validation_results[
    "model_b_residual"
] = (
    validation_results[
        "actual_sale_price"
    ]
    -
    validation_results[
        "model_b_prediction"
    ]
)


validation_results[
    "model_a_absolute_error"
] = np.abs(
    validation_results[
        "model_a_residual"
    ]
)


validation_results[
    "model_b_absolute_error"
] = np.abs(
    validation_results[
        "model_b_residual"
    ]
)


validation_results.to_csv(
    TABLES_DIR
    / "house_prices_required_validation_predictions.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_validation_predictions.csv"
)


# ---------------------------------------------------------
# Residual summary
# ---------------------------------------------------------

model_a_residual = (
    validation_results[
        "model_a_residual"
    ]
)


model_b_residual = (
    validation_results[
        "model_b_residual"
    ]
)


residual_diagnostics = pd.DataFrame([
    {
        "model":
            "Model A - Engineered Linear Regression",

        "mean_residual":
            float(model_a_residual.mean()),

        "median_residual":
            float(model_a_residual.median()),

        "std_residual":
            float(model_a_residual.std()),

        "minimum_residual":
            float(model_a_residual.min()),

        "maximum_residual":
            float(model_a_residual.max()),

        "mean_absolute_residual":
            float(
                np.mean(
                    np.abs(
                        model_a_residual
                    )
                )
            )
    },

    {
        "model":
            "Model B - PCA Ridge Regression",

        "mean_residual":
            float(model_b_residual.mean()),

        "median_residual":
            float(model_b_residual.median()),

        "std_residual":
            float(model_b_residual.std()),

        "minimum_residual":
            float(model_b_residual.min()),

        "maximum_residual":
            float(model_b_residual.max()),

        "mean_absolute_residual":
            float(
                np.mean(
                    np.abs(
                        model_b_residual
                    )
                )
            )
    }
])


residual_diagnostics.to_csv(
    TABLES_DIR
    / "house_prices_required_residual_diagnostics.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_residual_diagnostics.csv"
)


# =========================================================
# Assumption Investigation
# =========================================================

print(
    "\nInvestigating regression assumptions..."
)


# ---------------------------------------------------------
# Assumption 1: residual centering
# ---------------------------------------------------------

assumption_results = []


for model_name, residual_series in [
    (
        "Model A - Engineered Linear Regression",
        model_a_residual
    ),
    (
        "Model B - PCA Ridge Regression",
        model_b_residual
    )
]:

    mean_residual = (
        float(
            residual_series.mean()
        )
    )

    std_residual = (
        float(
            residual_series.std()
        )
    )

    standardized_mean_residual = (
        mean_residual
        /
        std_residual
        if std_residual != 0
        else np.nan
    )

    assumption_results.append({
        "model":
            model_name,

        "diagnostic":
            "Residual centering",

        "value":
            mean_residual,

        "standardized_value":
            standardized_mean_residual,

        "interpretation_rule":
            "Residual mean should be near zero."
    })


# ---------------------------------------------------------
# Assumption 2: residual spread
# ---------------------------------------------------------

for model_name, residual_series in [
    (
        "Model A - Engineered Linear Regression",
        model_a_residual
    ),
    (
        "Model B - PCA Ridge Regression",
        model_b_residual
    )
]:

    q1 = float(
        residual_series.quantile(
            0.25
        )
    )

    q3 = float(
        residual_series.quantile(
            0.75
        )
    )

    iqr = (
        q3
        -
        q1
    )

    lower = (
        q1
        -
        1.5 * iqr
    )

    upper = (
        q3
        +
        1.5 * iqr
    )

    outlier_count = int(
        (
            (residual_series < lower)
            |
            (residual_series > upper)
        )
        .sum()
    )

    outlier_percentage = (
        100
        *
        outlier_count
        /
        len(residual_series)
    )

    assumption_results.append({
        "model":
            model_name,

        "diagnostic":
            "Residual outlier proportion",

        "value":
            float(outlier_percentage),

        "standardized_value":
            np.nan,

        "interpretation_rule":
            "A high proportion of extreme residuals may indicate influential observations or model misspecification."
    })


# ---------------------------------------------------------
# Create assumption table
# ---------------------------------------------------------

assumption_results_table = pd.DataFrame(
    assumption_results
)


assumption_results_table.to_csv(
    TABLES_DIR
    / "house_prices_required_assumption_diagnostics.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_assumption_diagnostics.csv"
)


# ---------------------------------------------------------
# Actual vs predicted plots
# ---------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)


# Model A

axes[0].scatter(
    validation_results[
        "actual_sale_price"
    ],
    validation_results[
        "model_a_prediction"
    ],
    alpha=0.45,
    s=25
)


a_min = min(
    validation_results[
        "actual_sale_price"
    ].min(),

    validation_results[
        "model_a_prediction"
    ].min()
)


a_max = max(
    validation_results[
        "actual_sale_price"
    ].max(),

    validation_results[
        "model_a_prediction"
    ].max()
)


axes[0].plot(
    [a_min, a_max],
    [a_min, a_max],
    color="red",
    linestyle="--",
    linewidth=2
)


axes[0].set_title(
    "Model A: Actual vs. Predicted"
)


axes[0].set_xlabel(
    "Actual SalePrice"
)


axes[0].set_ylabel(
    "Predicted SalePrice"
)


axes[0].grid(
    alpha=0.3
)


# Model B

axes[1].scatter(
    validation_results[
        "actual_sale_price"
    ],
    validation_results[
        "model_b_prediction"
    ],
    alpha=0.45,
    s=25
)


b_min = min(
    validation_results[
        "actual_sale_price"
    ].min(),

    validation_results[
        "model_b_prediction"
    ].min()
)


b_max = max(
    validation_results[
        "actual_sale_price"
    ].max(),

    validation_results[
        "model_b_prediction"
    ].max()
)


axes[1].plot(
    [b_min, b_max],
    [b_min, b_max],
    color="red",
    linestyle="--",
    linewidth=2
)


axes[1].set_title(
    "Model B: Actual vs. Predicted"
)


axes[1].set_xlabel(
    "Actual SalePrice"
)


axes[1].set_ylabel(
    "Predicted SalePrice"
)


axes[1].grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    FIGURES_DIR
    / "house_prices_required_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved figure:",
    FIGURES_DIR
    / "house_prices_required_actual_vs_predicted.png"
)


# ---------------------------------------------------------
# Residual plots
# ---------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)


axes[0].scatter(
    validation_results[
        "model_a_prediction"
    ],
    validation_results[
        "model_a_residual"
    ],
    alpha=0.45,
    s=25
)


axes[0].axhline(
    0,
    color="red",
    linestyle="--",
    linewidth=2
)


axes[0].set_title(
    "Model A: Residuals vs. Predicted"
)


axes[0].set_xlabel(
    "Predicted SalePrice"
)


axes[0].set_ylabel(
    "Residual"
)


axes[0].grid(
    alpha=0.3
)


axes[1].scatter(
    validation_results[
        "model_b_prediction"
    ],
    validation_results[
        "model_b_residual"
    ],
    alpha=0.45,
    s=25
)


axes[1].axhline(
    0,
    color="red",
    linestyle="--",
    linewidth=2
)


axes[1].set_title(
    "Model B: Residuals vs. Predicted"
)


axes[1].set_xlabel(
    "Predicted SalePrice"
)


axes[1].set_ylabel(
    "Residual"
)


axes[1].grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    FIGURES_DIR
    / "house_prices_required_residual_diagnostics.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved figure:",
    FIGURES_DIR
    / "house_prices_required_residual_diagnostics.png"
)


# ---------------------------------------------------------
# Residual distributions
# ---------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)


sns.histplot(
    model_a_residual,
    bins=35,
    kde=True,
    ax=axes[0],
    color="steelblue"
)


axes[0].set_title(
    "Model A Residual Distribution"
)


axes[0].set_xlabel(
    "Residual"
)


axes[0].grid(
    axis="y",
    alpha=0.3
)


sns.histplot(
    model_b_residual,
    bins=35,
    kde=True,
    ax=axes[1],
    color="darkorange"
)


axes[1].set_title(
    "Model B Residual Distribution"
)


axes[1].set_xlabel(
    "Residual"
)


axes[1].grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    FIGURES_DIR
    / "house_prices_required_residual_distributions.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved figure:",
    FIGURES_DIR
    / "house_prices_required_residual_distributions.png"
)


# =========================================================
# Final Model Refit
# =========================================================

print(
    "\n============================================================"
)

print(
    "FINAL HOUSE PRICES REQUIRED MODELS"
)

print(
    "============================================================"
)


# ---------------------------------------------------------
# Refit Model A using all training observations
# ---------------------------------------------------------

print(
    "\nRefitting Model A using all available training data..."
)


final_model_a = Pipeline(
    steps=[
        (
            "preprocessor",
            encoded_preprocessor
        ),

        (
            "subset_selection",
            SelectKBest(
                score_func=f_regression,
                k=SELECT_K_BEST
            )
        ),

        (
            "regression",
            LinearRegression()
        )
    ]
)


final_model_a.fit(
    X_engineered,
    y_log
)


print(
    "Final Model A fitting completed."
)


# ---------------------------------------------------------
# Refit Model B using all training observations
# ---------------------------------------------------------

print(
    "\nRefitting Model B using all available training data..."
)


final_model_b = Pipeline(
    steps=[
        (
            "preprocessor",
            dense_encoded_preprocessor
        ),

        (
            "pca",
            PCA(
                n_components=PCA_VARIANCE,
                svd_solver="full",
                random_state=RANDOM_STATE
            )
        ),

        (
            "regression",
            Ridge(
                alpha=RIDGE_ALPHA
            )
        )
    ]
)


final_model_b.fit(
    X_engineered,
    y_log
)


print(
    "Final Model B fitting completed."
)


# =========================================================
# Generate Test Predictions
# =========================================================

print(
    "\nGenerating final House Prices test predictions..."
)


final_model_a_log_predictions = (
    final_model_a.predict(
        X_test_engineered
    )
)


final_model_b_log_predictions = (
    final_model_b.predict(
        X_test_engineered
    )
)


final_model_a_predictions = (
    np.maximum(
        np.expm1(
            final_model_a_log_predictions
        ),
        0
    )
)


final_model_b_predictions = (
    np.maximum(
        np.expm1(
            final_model_b_log_predictions
        ),
        0
    )
)


print(
    "Model A test predictions generated:",
    len(final_model_a_predictions)
)


print(
    "Model B test predictions generated:",
    len(final_model_b_predictions)
)


# =========================================================
# Create Kaggle-ready submission files
# =========================================================

model_a_submission = pd.DataFrame({
    "Id":
        test_ids,

    "SalePrice":
        final_model_a_predictions
})


model_a_submission.to_csv(
    SUBMISSIONS_DIR
    / "house_prices_model_a_submission.csv",
    index=False
)


print(
    "Saved submission:",
    SUBMISSIONS_DIR
    / "house_prices_model_a_submission.csv"
)


model_b_submission = pd.DataFrame({
    "Id":
        test_ids,

    "SalePrice":
        final_model_b_predictions
})


model_b_submission.to_csv(
    SUBMISSIONS_DIR
    / "house_prices_model_b_submission.csv",
    index=False
)


print(
    "Saved submission:",
    SUBMISSIONS_DIR
    / "house_prices_model_b_submission.csv"
)


# ---------------------------------------------------------
# Report-ready prediction table
# ---------------------------------------------------------

test_prediction_table = pd.DataFrame({
    "Id":
        test_ids,

    "model_a_predicted_SalePrice":
        final_model_a_predictions,

    "model_b_predicted_SalePrice":
        final_model_b_predictions
})


test_prediction_table.to_csv(
    TABLES_DIR
    / "house_prices_required_test_predictions.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_test_predictions.csv"
)


# =========================================================
# Test Prediction Summary
# =========================================================

test_prediction_summary = pd.DataFrame([
    {
        "model":
            "Model A - Engineered Linear Regression",

        "observations":
            len(final_model_a_predictions),

        "mean":
            float(
                np.mean(
                    final_model_a_predictions
                )
            ),

        "median":
            float(
                np.median(
                    final_model_a_predictions
                )
            ),

        "minimum":
            float(
                np.min(
                    final_model_a_predictions
                )
            ),

        "maximum":
            float(
                np.max(
                    final_model_a_predictions
                )
            ),

        "standard_deviation":
            float(
                np.std(
                    final_model_a_predictions
                )
            )
    },

    {
        "model":
            "Model B - PCA Ridge Regression",

        "observations":
            len(final_model_b_predictions),

        "mean":
            float(
                np.mean(
                    final_model_b_predictions
                )
            ),

        "median":
            float(
                np.median(
                    final_model_b_predictions
                )
            ),

        "minimum":
            float(
                np.min(
                    final_model_b_predictions
                )
            ),

        "maximum":
            float(
                np.max(
                    final_model_b_predictions
                )
            ),

        "standard_deviation":
            float(
                np.std(
                    final_model_b_predictions
                )
            )
    }
])


test_prediction_summary.to_csv(
    TABLES_DIR
    / "house_prices_required_test_prediction_summary.csv",
    index=False
)


print(
    "Saved table:",
    TABLES_DIR
    / "house_prices_required_test_prediction_summary.csv"
)


# =========================================================
# Test Prediction Distribution
# =========================================================

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    final_model_a_predictions,
    bins=40,
    alpha=0.55,
    label="Model A",
    color="steelblue"
)


plt.hist(
    final_model_b_predictions,
    bins=40,
    alpha=0.55,
    label="Model B",
    color="darkorange"
)


plt.xlabel(
    "Predicted SalePrice"
)


plt.ylabel(
    "Number of Test Observations"
)


plt.title(
    "House Prices Required Models: Test Prediction Distributions"
)


plt.legend()


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    FIGURES_DIR
    / "house_prices_required_test_prediction_distribution.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved figure:",
    FIGURES_DIR
    / "house_prices_required_test_prediction_distribution.png"
)


# =========================================================
# Final Results JSON
# =========================================================

house_prices_required_results = {

    "project":
        "Assignment 8",

    "dataset":
        "House Prices: Advanced Regression Techniques",

    "objective":
        "Address required regression techniques",

    "target":
        TARGET_COLUMN,

    "response_transformation":
        "log1p(SalePrice)",

    "validation": {

        "validation_size":
            float(
                VALIDATION_SIZE
            ),

        "random_state":
            int(
                RANDOM_STATE
            ),

        "training_observations":
            int(
                len(X_train)
            ),

        "validation_observations":
            int(
                len(X_validation)
            )
    },

    "required_techniques": {

        "dichotomous_variables":
            "Categorical variables represented through one-hot encoding",

        "interaction_terms": [
            "OverallQual_x_GrLivArea",
            "OverallQual_x_TotalBsmtSF",
            "GrLivArea_x_GarageCars"
        ],

        "polynomial_terms": [
            "OverallQual_squared",
            "GrLivArea_squared",
            "TotalBsmtSF_squared"
        ],

        "subset_investigation":
            f"SelectKBest with k={SELECT_K_BEST}",

        "transformation":
            "log1p(SalePrice)",

        "dimension_reduction":
            f"PCA retaining approximately {PCA_VARIANCE:.0%} variance"
    },

    "model_a": {

        "name":
            "Engineered Linear Regression",

        "MAE":
            model_a_original_metrics["MAE"],

        "RMSE":
            model_a_original_metrics["RMSE"],

        "R2":
            model_a_original_metrics["R2"],

        "log_MAE":
            model_a_log_metrics["log_MAE"],

        "log_RMSE":
            model_a_log_metrics["log_RMSE"],

        "log_R2":
            model_a_log_metrics["log_R2"],

        "selected_predictors":
            int(
                len(
                    model_a_selected_features
                )
            )
    },

    "model_b": {

        "name":
            "PCA Ridge Regression",

        "MAE":
            model_b_original_metrics["MAE"],

        "RMSE":
            model_b_original_metrics["RMSE"],

        "R2":
            model_b_original_metrics["R2"],

        "log_MAE":
            model_b_log_metrics["log_MAE"],

        "log_RMSE":
            model_b_log_metrics["log_RMSE"],

        "log_R2":
            model_b_log_metrics["log_R2"],

        "pca_components":
            int(
                fitted_pca.n_components_
            ),

        "pca_explained_variance":
            float(
                cumulative_explained_variance[-1]
            ),

        "ridge_alpha":
            float(
                RIDGE_ALPHA
            )
    },

    "validation_model_selection": {

        "criterion":
            "Validation Log RMSE",

        "selected_model":
            selected_model_name,

        "selected_log_RMSE":
            float(
                selected_validation_log_rmse
            ),

        "selected_original_RMSE":
            float(
                selected_validation_rmse
            ),

        "selected_original_MAE":
            float(
                selected_validation_mae
            ),

        "selected_original_R2":
            float(
                selected_validation_r2
            )
    },

    "test": {

        "observations":
            int(
                len(
                    final_model_a_predictions
                )
            ),

        "model_a_mean_prediction":
            float(
                np.mean(
                    final_model_a_predictions
                )
            ),

        "model_b_mean_prediction":
            float(
                np.mean(
                    final_model_b_predictions
                )
            )
    }
}


results_path = (
    RESULTS_DIR
    / "house_prices_required_techniques_results.json"
)


with open(
    results_path,
    "w",
    encoding="utf-8"
) as results_file:

    json.dump(
        house_prices_required_results,
        results_file,
        indent=4
    )


print(
    "Saved results:",
    results_path
)


# =========================================================
# Final Artifact Inventory
# =========================================================

print(
    "\nFinal House Prices Required Techniques artifact inventory:"
)


print(
    "  Model A:",
    "Engineered Linear Regression"
)


print(
    "  Model B:",
    "PCA Ridge Regression"
)


print(
    "  Validation selection criterion:",
    "Log RMSE"
)


print(
    "  Model A validation Log RMSE:",
    model_a_log_metrics["log_RMSE"]
)


print(
    "  Model B validation Log RMSE:",
    model_b_log_metrics["log_RMSE"]
)


print(
    "  Selected validation model:",
    selected_model_name
)


print(
    "  PCA components:",
    fitted_pca.n_components_
)


print(
    "  PCA explained variance:",
    cumulative_explained_variance[-1]
)


print(
    "  Model A selected predictors:",
    len(model_a_selected_features)
)


print(
    "  Test prediction rows:",
    len(final_model_a_predictions)
)


print(
    "  Model A submission:",
    SUBMISSIONS_DIR
    / "house_prices_model_a_submission.csv"
)


print(
    "  Model B submission:",
    SUBMISSIONS_DIR
    / "house_prices_model_b_submission.csv"
)


print(
    "\n" + "=" * 60
)


print(
    "10_house_prices_required_techniques.py completed successfully."
)


print(
    "=" * 60
)


