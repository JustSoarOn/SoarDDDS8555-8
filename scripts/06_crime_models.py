# ============================================================
# Assignment 8: Build & Evaluate
# Time Series, Regression, and Classification Models
#
# 06_crime_models.py
#
# Purpose:
#   Build and evaluate classification models for the
#   San Francisco Crime Classification dataset.
#
# Focus:
#   - Data inspection
#   - Target and feature preparation
#   - Train/validation split
#   - Baseline classification
#   - Logistic regression
#   - Tree-based classification
#   - Report-ready evaluation artifacts
#
# Report workflow:
#   This script creates durable tables, figures, and JSON
#   results for interpretation and presentation in
#   SoarDDDS8555-8-Report.Rmd.
# ============================================================


from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------
# Project directories
# ---------------------------------------------------------

PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
)

DATA_DIR = PROJECT_DIR / "data"

CRIME_DIR = (
    DATA_DIR
    / "san_francisco_crime_classification"
)


# ---------------------------------------------------------
# Output directories
# ---------------------------------------------------------

OUTPUT_DIR = PROJECT_DIR / "output"

FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = OUTPUT_DIR / "results"

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


# ---------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------

VALIDATION_SIZE = 0.20
RANDOM_STATE = 8555

DECISION_TREE_MAX_DEPTH = 12
DECISION_TREE_MIN_SAMPLES_LEAF = 10

LOGISTIC_REGRESSION_MAX_ITERATIONS = 1000


# ---------------------------------------------------------
# Start
# ---------------------------------------------------------

print("=" * 60)
print("06_crime_models.py")
print("=" * 60)

print(
    "\nSan Francisco Crime Classification data directory:"
)

print(CRIME_DIR)


# ---------------------------------------------------------
# Verify expected files
# ---------------------------------------------------------

expected_files = [
    "train.csv",
    "test.csv",
    "sample-submission.csv"
]

print("\nExpected active data files:")

for expected_file_name in expected_files:

    expected_file_path = (
        CRIME_DIR / expected_file_name
    )

    print(
        f"  {expected_file_name}: "
        f"{expected_file_path.exists()}"
    )


# ---------------------------------------------------------
# Load training data
# ---------------------------------------------------------

print("\nLoading crime training data...")

crime_training_data = pd.read_csv(
    CRIME_DIR / "train.csv"
)

print(
    "Training data shape:",
    crime_training_data.shape
)

print("\nTraining columns:")

print(
    crime_training_data.columns.tolist()
)


# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------

print("\nLoading crime test data...")

crime_test_data = pd.read_csv(
    CRIME_DIR / "test.csv"
)

print(
    "Test data shape:",
    crime_test_data.shape
)

print("\nTest columns:")

print(
    crime_test_data.columns.tolist()
)


# ---------------------------------------------------------
# Load sample submission
# ---------------------------------------------------------

print("\nLoading sample submission...")

crime_sample_submission = pd.read_csv(
    CRIME_DIR / "sample-submission.csv"
)

print(
    "Sample submission shape:",
    crime_sample_submission.shape
)

print("\nSample submission columns:")

print(
    crime_sample_submission.columns.tolist()
)


# ---------------------------------------------------------
# Basic dataset inspection
# ---------------------------------------------------------

print("\nTraining data information:")

print(
    crime_training_data.info()
)

print("\nMissing values in training data:")

print(
    crime_training_data.isna().sum()
)


# ---------------------------------------------------------
# Identify target column
# ---------------------------------------------------------

target_column_candidates = [
    "Category"
]

available_target_columns = [
    column_name
    for column_name in target_column_candidates
    if column_name in crime_training_data.columns
]

if len(available_target_columns) != 1:

    raise ValueError(
        "Expected target column 'Category' "
        "was not found exactly once."
    )

crime_target_column = (
    available_target_columns[0]
)


print(
    "\nClassification target:",
    crime_target_column
)


# ---------------------------------------------------------
# Target distribution
# ---------------------------------------------------------

crime_target_distribution = (
    crime_training_data[
        crime_target_column
    ]
    .value_counts()
    .rename_axis(
        "crime_category"
    )
    .reset_index(
        name="observation_count"
    )
)

crime_target_distribution[
    "percentage"
] = (
    crime_target_distribution[
        "observation_count"
    ]
    / len(crime_training_data)
    * 100
)


crime_target_distribution.to_csv(
    TABLES_DIR
    / "crime_target_distribution.csv",
    index=False
)

print(
    "Saved table:",
    TABLES_DIR
    / "crime_target_distribution.csv"
)


# ---------------------------------------------------------
# Target distribution figure
# ---------------------------------------------------------

plt.figure(
    figsize=(12, 8)
)

sns.barplot(
    data=crime_target_distribution,
    x="observation_count",
    y="crime_category",
    color="steelblue"
)

plt.title(
    "San Francisco Crime Category Distribution"
)

plt.xlabel(
    "Number of Training Observations"
)

plt.ylabel(
    "Crime Category"
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "crime_target_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figure:",
    FIGURES_DIR
    / "crime_target_distribution.png"
)


# ---------------------------------------------------------
# Identify identifier columns
# ---------------------------------------------------------

identifier_columns = []

for possible_identifier in [
    "Dates",
    "Descript",
    "Resolution",
    "Id"
]:

    if possible_identifier in crime_training_data.columns:

        identifier_columns.append(
            possible_identifier
        )


print(
    "\nIdentifier / non-predictive columns:",
    identifier_columns
)


# ---------------------------------------------------------
# Feature engineering from Dates
# ---------------------------------------------------------

crime_training_features = (
    crime_training_data.copy()
)

crime_test_features = (
    crime_test_data.copy()
)


if "Dates" in crime_training_features.columns:

    crime_training_features["Dates"] = (
        pd.to_datetime(
            crime_training_features["Dates"]
        )
    )

    crime_test_features["Dates"] = (
        pd.to_datetime(
            crime_test_features["Dates"]
        )
    )

    for feature_name, date_component in [
        (
            "year",
            crime_training_features[
                "Dates"
            ].dt.year
        ),
        (
            "month",
            crime_training_features[
                "Dates"
            ].dt.month
        ),
        (
            "day",
            crime_training_features[
                "Dates"
            ].dt.day
        ),
        (
            "hour",
            crime_training_features[
                "Dates"
            ].dt.hour
        ),
        (
            "day_of_week",
            crime_training_features[
                "Dates"
            ].dt.dayofweek
        )
    ]:

        crime_training_features[
            f"date_{feature_name}"
        ] = date_component


    crime_test_features[
        "date_year"
    ] = crime_test_features[
        "Dates"
    ].dt.year

    crime_test_features[
        "date_month"
    ] = crime_test_features[
        "Dates"
    ].dt.month

    crime_test_features[
        "date_day"
    ] = crime_test_features[
        "Dates"
    ].dt.day

    crime_test_features[
        "date_hour"
    ] = crime_test_features[
        "Dates"
    ].dt.hour

    crime_test_features[
        "date_day_of_week"
    ] = crime_test_features[
        "Dates"
    ].dt.dayofweek


# ---------------------------------------------------------
# Remove target and non-predictive text columns
# ---------------------------------------------------------

columns_to_exclude_from_model = [
    crime_target_column,
    "Id",
    "Dates",
    "Address",
    "Descript",
    "Resolution"
]


model_feature_columns = [
    column_name
    for column_name in crime_training_features.columns
    if column_name not in
    columns_to_exclude_from_model
]


print(
    "\nModel feature columns:"
)

print(
    model_feature_columns
)


# ---------------------------------------------------------
# Create modeling data
# ---------------------------------------------------------

crime_model_features = (
    crime_training_features[
        model_feature_columns
    ]
    .copy()
)

crime_model_target = (
    crime_training_features[
        crime_target_column
    ]
    .copy()
)


crime_test_model_features = (
    crime_test_features[
        model_feature_columns
    ]
    .copy()
)


# ---------------------------------------------------------
# Identify numerical and categorical features
# ---------------------------------------------------------

numerical_feature_columns = (
    crime_model_features
    .select_dtypes(
        include=[
            "number",
            "bool"
        ]
    )
    .columns
    .tolist()
)

categorical_feature_columns = (
    crime_model_features
    .select_dtypes(
        include=[
            "object",
            "category"
        ]
    )
    .columns
    .tolist()
)


print(
    "\nNumerical feature count:",
    len(numerical_feature_columns)
)

print(
    "Categorical feature count:",
    len(categorical_feature_columns)
)


print(
    "\nCategorical features:",
    categorical_feature_columns
)


# ---------------------------------------------------------
# Save feature metadata
# ---------------------------------------------------------

crime_feature_metadata = {

    "target_column":
        crime_target_column,

    "training_rows":
        int(len(crime_training_data)),

    "test_rows":
        int(len(crime_test_data)),

    "model_feature_count":
        int(len(model_feature_columns)),

    "numerical_feature_count":
        int(len(numerical_feature_columns)),

    "categorical_feature_count":
        int(len(categorical_feature_columns)),

    "model_features":
        model_feature_columns,

    "numerical_features":
        numerical_feature_columns,

    "categorical_features":
        categorical_feature_columns,

    "excluded_columns":
        columns_to_exclude_from_model
}


with open(
    RESULTS_DIR
    / "crime_feature_metadata.json",
    "w",
    encoding="utf-8"
) as metadata_file:

    json.dump(
        crime_feature_metadata,
        metadata_file,
        indent=4
    )


print(
    "Saved results:",
    RESULTS_DIR
    / "crime_feature_metadata.json"
)


# ---------------------------------------------------------
# Train/validation split
# ---------------------------------------------------------

(
    crime_training_features_split,
    crime_validation_features_split,
    crime_training_target_split,
    crime_validation_target_split
) = train_test_split(

    crime_model_features,

    crime_model_target,

    test_size=VALIDATION_SIZE,

    random_state=RANDOM_STATE,

    stratify=crime_model_target
)


print(
    "\nClassification validation split"
)

print(
    "Training observations:",
    len(crime_training_features_split)
)

print(
    "Validation observations:",
    len(crime_validation_features_split)
)


# ---------------------------------------------------------
# Preprocessing pipelines
# ---------------------------------------------------------

numerical_preprocessor = Pipeline(
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


categorical_preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


crime_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_preprocessor,
            numerical_feature_columns
        ),
        (
            "categorical",
            categorical_preprocessor,
            categorical_feature_columns
        )
    ]
)


# =========================================================
# BASELINE CLASSIFIER
# =========================================================

crime_baseline_classifier = Pipeline(
    steps=[
        (
            "preprocessor",
            crime_preprocessor
        ),
        (
            "classifier",
            DummyClassifier(
                strategy="most_frequent"
            )
        )
    ]
)


crime_baseline_classifier.fit(
    crime_training_features_split,
    crime_training_target_split
)


crime_baseline_predictions = (
    crime_baseline_classifier.predict(
        crime_validation_features_split
    )
)


crime_baseline_probabilities = (
    crime_baseline_classifier.predict_proba(
        crime_validation_features_split
    )
)


crime_baseline_accuracy = (
    accuracy_score(
        crime_validation_target_split,
        crime_baseline_predictions
    )
)


crime_baseline_log_loss = (
    log_loss(
        crime_validation_target_split,
        crime_baseline_probabilities,
        labels=(
            crime_baseline_classifier[
                "classifier"
            ].classes_
        )
    )
)


print(
    "\nBaseline classifier:"
)

print(
    "Validation accuracy:",
    crime_baseline_accuracy
)

print(
    "Validation log loss:",
    crime_baseline_log_loss
)


# ---------------------------------------------------------
# Prepare models for next section
# ---------------------------------------------------------

crime_logistic_regression_classifier = Pipeline(
    steps=[
        (
            "preprocessor",
            crime_preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=
                    LOGISTIC_REGRESSION_MAX_ITERATIONS,
                solver="lbfgs"
            )
        )
    ]
)


crime_decision_tree_classifier = Pipeline(
    steps=[
        (
            "preprocessor",
            crime_preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=
                    DECISION_TREE_MAX_DEPTH,
                min_samples_leaf=
                    DECISION_TREE_MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ---------------------------------------------------------
# Durable split summary
# ---------------------------------------------------------

crime_validation_split_summary = pd.DataFrame({

    "dataset": [
        "model_training",
        "validation"
    ],

    "observations": [
        len(
            crime_training_features_split
        ),
        len(
            crime_validation_features_split
        )
    ],

    "percentage": [
        len(
            crime_training_features_split
        )
        / len(crime_model_features)
        * 100,

        len(
            crime_validation_features_split
        )
        / len(crime_model_features)
        * 100
    ]
})


crime_validation_split_summary.to_csv(
    TABLES_DIR
    / "crime_validation_split_summary.csv",
    index=False
)


# ---------------------------------------------------------
# Baseline results
# ---------------------------------------------------------

crime_baseline_results = pd.DataFrame({

    "model": [
        "Most Frequent Baseline"
    ],

    "accuracy": [
        crime_baseline_accuracy
    ],

    "log_loss": [
        crime_baseline_log_loss
    ]
})


crime_baseline_results.to_csv(
    TABLES_DIR
    / "crime_baseline_evaluation.csv",
    index=False
)


print(
    "\nSaved table:",
    TABLES_DIR
    / "crime_validation_split_summary.csv"
)

print(
    "Saved table:",
    TABLES_DIR
    / "crime_baseline_evaluation.csv"
)


print(
    "\n06_crime_models.py data preparation "
    "and baseline modeling completed."
)

print(
    "Ready for the classification model evaluation section."
)

# =========================================================
# Classification Model Evaluation
# =========================================================

print("\nClassification model evaluation")
print("-" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# Prepare model evaluation data
# ---------------------------------------------------------

crime_model_training_features = (
    crime_training_features_split.copy()
)

crime_model_validation_features = (
    crime_validation_features_split.copy()
)

crime_model_training_target = (
    crime_training_target_split.copy()
)

crime_model_validation_target = (
    crime_validation_target_split.copy()
)


# ---------------------------------------------------------
# Logistic Regression Classifier
# ---------------------------------------------------------

crime_logistic_regression_classifier.fit(
    crime_model_training_features,
    crime_model_training_target
)


crime_logistic_regression_predictions = (
    crime_logistic_regression_classifier.predict(
        crime_model_validation_features
    )
)


crime_logistic_regression_probabilities = (
    crime_logistic_regression_classifier.predict_proba(
        crime_model_validation_features
    )
)


crime_logistic_regression_accuracy = (
    accuracy_score(
        crime_model_validation_target,
        crime_logistic_regression_predictions
    )
)


crime_logistic_regression_log_loss = (
    log_loss(
        crime_model_validation_target,
        crime_logistic_regression_probabilities,
        labels=(
            crime_logistic_regression_classifier[
                "classifier"
            ].classes_
        )
    )
)


print(
    "\nLogistic Regression:"
)

print(
    "Validation accuracy:",
    crime_logistic_regression_accuracy
)

print(
    "Validation log loss:",
    crime_logistic_regression_log_loss
)


# ---------------------------------------------------------
# Logistic Regression Classification Report
# ---------------------------------------------------------

crime_logistic_regression_report = (
    classification_report(
        crime_model_validation_target,
        crime_logistic_regression_predictions,
        output_dict=True,
        zero_division=0
    )
)


crime_logistic_regression_report_table = (
    pd.DataFrame(
        crime_logistic_regression_report
    )
    .transpose()
    .reset_index()
    .rename(
        columns={
            "index": "class"
        }
    )
)


crime_logistic_regression_report_table.to_csv(
    TABLES_DIR
    / "crime_logistic_regression_classification_report.csv",
    index=False
)


# ---------------------------------------------------------
# Logistic Regression Confusion Matrix
# ---------------------------------------------------------

crime_logistic_regression_confusion_matrix = (
    confusion_matrix(
        crime_model_validation_target,
        crime_logistic_regression_predictions,
        labels=(
            crime_logistic_regression_classifier[
                "classifier"
            ].classes_
        )
    )
)


crime_logistic_regression_confusion_matrix_table = (
    pd.DataFrame(
        crime_logistic_regression_confusion_matrix,
        index=(
            crime_logistic_regression_classifier[
                "classifier"
            ].classes_
        ),
        columns=(
            crime_logistic_regression_classifier[
                "classifier"
            ].classes_
        )
    )
)


crime_logistic_regression_confusion_matrix_table.to_csv(
    TABLES_DIR
    / "crime_logistic_regression_confusion_matrix.csv"
)


# ---------------------------------------------------------
# Decision Tree Classifier
# ---------------------------------------------------------

crime_decision_tree_classifier.fit(
    crime_model_training_features,
    crime_model_training_target
)


crime_decision_tree_predictions = (
    crime_decision_tree_classifier.predict(
        crime_model_validation_features
    )
)


crime_decision_tree_probabilities = (
    crime_decision_tree_classifier.predict_proba(
        crime_model_validation_features
    )
)


crime_decision_tree_accuracy = (
    accuracy_score(
        crime_model_validation_target,
        crime_decision_tree_predictions
    )
)


crime_decision_tree_log_loss = (
    log_loss(
        crime_model_validation_target,
        crime_decision_tree_probabilities,
        labels=(
            crime_decision_tree_classifier[
                "classifier"
            ].classes_
        )
    )
)


print(
    "\nDecision Tree:"
)

print(
    "Validation accuracy:",
    crime_decision_tree_accuracy
)

print(
    "Validation log loss:",
    crime_decision_tree_log_loss
)


# ---------------------------------------------------------
# Decision Tree Classification Report
# ---------------------------------------------------------

crime_decision_tree_report = (
    classification_report(
        crime_model_validation_target,
        crime_decision_tree_predictions,
        output_dict=True,
        zero_division=0
    )
)


crime_decision_tree_report_table = (
    pd.DataFrame(
        crime_decision_tree_report
    )
    .transpose()
    .reset_index()
    .rename(
        columns={
            "index": "class"
        }
    )
)


crime_decision_tree_report_table.to_csv(
    TABLES_DIR
    / "crime_decision_tree_classification_report.csv",
    index=False
)


# ---------------------------------------------------------
# Decision Tree Confusion Matrix
# ---------------------------------------------------------

crime_decision_tree_confusion_matrix = (
    confusion_matrix(
        crime_model_validation_target,
        crime_decision_tree_predictions,
        labels=(
            crime_decision_tree_classifier[
                "classifier"
            ].classes_
        )
    )
)


crime_decision_tree_confusion_matrix_table = (
    pd.DataFrame(
        crime_decision_tree_confusion_matrix,
        index=(
            crime_decision_tree_classifier[
                "classifier"
            ].classes_
        ),
        columns=(
            crime_decision_tree_classifier[
                "classifier"
            ].classes_
        )
    )
)


crime_decision_tree_confusion_matrix_table.to_csv(
    TABLES_DIR
    / "crime_decision_tree_confusion_matrix.csv"
)


# ---------------------------------------------------------
# Model evaluation summary
# ---------------------------------------------------------

crime_model_evaluation_results = pd.DataFrame({

    "model": [
        "Most Frequent Baseline",
        "Logistic Regression",
        "Decision Tree"
    ],

    "accuracy": [
        crime_baseline_accuracy,
        crime_logistic_regression_accuracy,
        crime_decision_tree_accuracy
    ],

    "log_loss": [
        crime_baseline_log_loss,
        crime_logistic_regression_log_loss,
        crime_decision_tree_log_loss
    ]
})


crime_model_evaluation_results.to_csv(
    TABLES_DIR
    / "crime_model_evaluation.csv",
    index=False
)


# ---------------------------------------------------------
# Save model evaluation results
# ---------------------------------------------------------

crime_model_evaluation_results_json = {
    "validation_observations": int(
        len(crime_model_validation_target)
    ),

    "models": (
        crime_model_evaluation_results
        .to_dict(orient="records")
    )
}


with open(
    RESULTS_DIR
    / "crime_model_evaluation_results.json",
    "w",
    encoding="utf-8"
) as results_file:

    json.dump(
        crime_model_evaluation_results_json,
        results_file,
        indent=4
    )


# ---------------------------------------------------------
# Completion
# ---------------------------------------------------------

print(
    "\nSaved table:",
    TABLES_DIR
    / "crime_model_evaluation.csv"
)

print(
    "Saved results:",
    RESULTS_DIR
    / "crime_model_evaluation_results.json"
)

print("\n" + "=" * 60)
print(
    "\n06_crime_models.py classification model evaluation "
    "completed."
)
print("=" * 60)
