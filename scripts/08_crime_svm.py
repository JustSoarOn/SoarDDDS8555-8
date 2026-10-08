# ============================================================
# Assignment 8: Build & Evaluate
# Time Series, Regression, and Classification Models
#
# 08_crime_svm.py
#
# Purpose:
#   Build and evaluate the required Support Vector Machine
#   classification model for the San Francisco Crime
#   Classification dataset.
#
# Focus:
#   - Independent data loading
#   - Consistent feature engineering
#   - Reproducible validation split
#   - Support Vector Machine classification
#   - Validation performance evaluation
#   - Report-ready evaluation artifacts
#   - Kaggle-ready probability submission
#
# Model:
#   Linear Support Vector Machine (LinearSVC)
#
# Report workflow:
#   This script creates durable tables, figures, and JSON
#   results for interpretation and presentation in
#   SoarDDDS8555-8-Report.Rmd.
# ============================================================


# ============================================================
# Imports
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.svm import LinearSVC


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

CRIME_DIR = (
    DATA_DIR
    / "san_francisco_crime_classification"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
)

TABLES_DIR = (
    OUTPUT_DIR
    / "tables"
)

FIGURES_DIR = (
    OUTPUT_DIR
    / "figures"
)

RESULTS_DIR = (
    OUTPUT_DIR
    / "results"
)


TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Reproducibility settings
# ============================================================

RANDOM_STATE = 8555

VALIDATION_SIZE = 0.20

SVM_MAX_ITERATIONS = 2000

SVM_C = 1.0


# ============================================================
# Dataset paths
# ============================================================

CRIME_TRAIN_PATH = (
    CRIME_DIR
    / "train.csv"
)

CRIME_TEST_PATH = (
    CRIME_DIR
    / "test.csv"
)

CRIME_SAMPLE_SUBMISSION_PATH = (
    CRIME_DIR
    / "sample-submission.csv"
)


# ============================================================
# Verify required files
# ============================================================

required_crime_files = [
    CRIME_TRAIN_PATH,
    CRIME_TEST_PATH,
    CRIME_SAMPLE_SUBMISSION_PATH
]

missing_crime_files = [
    str(file_path)
    for file_path in required_crime_files
    if not file_path.exists()
]

if missing_crime_files:

    raise FileNotFoundError(
        "Required San Francisco Crime Classification "
        "files are missing:\n"
        + "\n".join(
            missing_crime_files
        )
    )


# ============================================================
# Load training data
# ============================================================

print("=" * 60)
print("08_crime_svm.py")
print("=" * 60)

print(
    "\nLoading San Francisco Crime Classification data..."
)

crime_training_data = pd.read_csv(
    CRIME_TRAIN_PATH
)

print(
    "Training data shape:",
    crime_training_data.shape
)


# ============================================================
# Load test data
# ============================================================

crime_test_data = pd.read_csv(
    CRIME_TEST_PATH
)

print(
    "Test data shape:",
    crime_test_data.shape
)


# ============================================================
# Load sample submission
# ============================================================

crime_sample_submission = pd.read_csv(
    CRIME_SAMPLE_SUBMISSION_PATH
)

print(
    "Sample submission shape:",
    crime_sample_submission.shape
)


# ============================================================
# Validate expected columns
# ============================================================

crime_target_column = "Category"

expected_training_columns = [
    "Dates",
    "Category",
    "Descript",
    "DayOfWeek",
    "PdDistrict",
    "Resolution",
    "Address",
    "X",
    "Y"
]

expected_test_columns = [
    "Id",
    "Dates",
    "DayOfWeek",
    "PdDistrict",
    "Address",
    "X",
    "Y"
]


missing_training_columns = [
    column_name
    for column_name in expected_training_columns
    if column_name not in crime_training_data.columns
]

missing_test_columns = [
    column_name
    for column_name in expected_test_columns
    if column_name not in crime_test_data.columns
]

if missing_training_columns:

    raise ValueError(
        "Training data is missing expected columns: "
        + str(missing_training_columns)
    )

if missing_test_columns:

    raise ValueError(
        "Test data is missing expected columns: "
        + str(missing_test_columns)
    )


# ============================================================
# Feature engineering
# ============================================================

print(
    "\nCreating crime classification model features..."
)

crime_training_features = (
    crime_training_data.copy()
)

crime_test_features = (
    crime_test_data.copy()
)


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
] = (
    crime_test_features[
        "Dates"
    ].dt.year
)

crime_test_features[
    "date_month"
] = (
    crime_test_features[
        "Dates"
    ].dt.month
)

crime_test_features[
    "date_day"
] = (
    crime_test_features[
        "Dates"
    ].dt.day
)

crime_test_features[
    "date_hour"
] = (
    crime_test_features[
        "Dates"
    ].dt.hour
)

crime_test_features[
    "date_day_of_week"
] = (
    crime_test_features[
        "Dates"
    ].dt.dayofweek
)


# ============================================================
# Define model features
# ============================================================

columns_to_exclude_from_model = [
    "Category",
    "Id",
    "Dates",
    "Descript",
    "Resolution"
]


model_feature_columns = [
    column_name
    for column_name in crime_training_features.columns
    if column_name
    not in columns_to_exclude_from_model
]


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


print(
    "\nModel feature columns:"
)

print(
    model_feature_columns
)


# ============================================================
# Identify feature types
# ============================================================

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
            "category",
            "str"
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
    "Categorical features:",
    categorical_feature_columns
)


# ============================================================
# Train/validation split
# ============================================================

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
    len(
        crime_training_features_split
    )
)

print(
    "Validation observations:",
    len(
        crime_validation_features_split
    )
)


# ============================================================
# Preprocessing pipelines
# ============================================================

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


crime_svm_preprocessor = ColumnTransformer(
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


# ============================================================
# SVM classifier
# ============================================================

crime_svm_classifier = Pipeline(
    steps=[
        (
            "preprocessor",
            crime_svm_preprocessor
        ),
        (
            "classifier",
            LinearSVC(
                C=SVM_C,
                max_iter=SVM_MAX_ITERATIONS,
                random_state=RANDOM_STATE
            )
        )
    ]
)


print(
    "\nSVM configuration:"
)

print(
    "Model: LinearSVC"
)

print(
    "C:",
    SVM_C
)

print(
    "Maximum iterations:",
    SVM_MAX_ITERATIONS
)

print(
    "\nReady to fit the Support Vector Machine."
)


# ============================================================
# Fit Support Vector Machine
# ============================================================

print(
    "\nFitting Support Vector Machine..."
)

crime_svm_classifier.fit(
    crime_training_features_split,
    crime_training_target_split
)

print(
    "Support Vector Machine fitting completed."
)


# ============================================================
# Validation predictions
# ============================================================

print(
    "\nGenerating SVM validation predictions..."
)

crime_svm_validation_predictions = (
    crime_svm_classifier.predict(
        crime_validation_features_split
    )
)


# LinearSVC does not provide predict_proba().
# Decision scores are retained for model diagnostics.
crime_svm_validation_decision_scores = (
    crime_svm_classifier.decision_function(
        crime_validation_features_split
    )
)


# ============================================================
# Validation metrics
# ============================================================

crime_svm_accuracy = accuracy_score(
    crime_validation_target_split,
    crime_svm_validation_predictions
)


crime_svm_macro_f1 = f1_score(
    crime_validation_target_split,
    crime_svm_validation_predictions,
    average="macro",
    zero_division=0
)


crime_svm_weighted_f1 = f1_score(
    crime_validation_target_split,
    crime_svm_validation_predictions,
    average="weighted",
    zero_division=0
)


print(
    "\nSupport Vector Machine validation performance:"
)

print(
    "Accuracy:",
    crime_svm_accuracy
)

print(
    "Macro F1:",
    crime_svm_macro_f1
)

print(
    "Weighted F1:",
    crime_svm_weighted_f1
)


# ============================================================
# Classification report
# ============================================================

crime_svm_classification_report = (
    classification_report(
        crime_validation_target_split,
        crime_svm_validation_predictions,
        output_dict=True,
        zero_division=0
    )
)


crime_svm_classification_report_table = (
    pd.DataFrame(
        crime_svm_classification_report
    )
    .transpose()
    .reset_index()
    .rename(
        columns={
            "index": "class"
        }
    )
)


crime_svm_classification_report_table.to_csv(
    TABLES_DIR
    / "crime_svm_classification_report.csv",
    index=False
)


# ============================================================
# Confusion matrix
# ============================================================

crime_svm_class_labels = (
    crime_svm_classifier[
        "classifier"
    ].classes_
)


crime_svm_confusion_matrix = (
    confusion_matrix(
        crime_validation_target_split,
        crime_svm_validation_predictions,
        labels=crime_svm_class_labels
    )
)


crime_svm_confusion_matrix_table = (
    pd.DataFrame(
        crime_svm_confusion_matrix,
        index=crime_svm_class_labels,
        columns=crime_svm_class_labels
    )
)


crime_svm_confusion_matrix_table.to_csv(
    TABLES_DIR
    / "crime_svm_confusion_matrix.csv"
)


# ============================================================
# Confusion matrix figure
# ============================================================

plt.figure(
    figsize=(16, 14)
)

sns.heatmap(
    crime_svm_confusion_matrix,
    cmap="Blues",
    xticklabels=crime_svm_class_labels,
    yticklabels=crime_svm_class_labels,
    cbar=True
)

plt.title(
    "Support Vector Machine Confusion Matrix"
)

plt.xlabel(
    "Predicted Crime Category"
)

plt.ylabel(
    "Actual Crime Category"
)

plt.xticks(
    rotation=90
)

plt.yticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "crime_svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SVM evaluation table
# ============================================================

crime_svm_evaluation = pd.DataFrame({

    "model": [
        "Linear Support Vector Machine"
    ],

    "accuracy": [
        crime_svm_accuracy
    ],

    "macro_f1": [
        crime_svm_macro_f1
    ],

    "weighted_f1": [
        crime_svm_weighted_f1
    ],

    "validation_observations": [
        len(
            crime_validation_target_split
        )
    ]
})


crime_svm_evaluation.to_csv(
    TABLES_DIR
    / "crime_svm_evaluation.csv",
    index=False
)


print(
    "\nSaved table:",
    TABLES_DIR
    / "crime_svm_evaluation.csv"
)


print(
    "Saved table:",
    TABLES_DIR
    / "crime_svm_classification_report.csv"
)

print(
    "Saved table:",
    TABLES_DIR
    / "crime_svm_confusion_matrix.csv"
)

print(
    "Saved figure:",
    FIGURES_DIR
    / "crime_svm_confusion_matrix.png"
)


# ============================================================
# SVM model metadata
# ============================================================

crime_svm_metadata = {

    "model":
        "Linear Support Vector Machine",

    "implementation":
        "sklearn.svm.LinearSVC",

    "kernel":
        "linear",

    "C":
        SVM_C,

    "max_iterations":
        SVM_MAX_ITERATIONS,

    "random_state":
        RANDOM_STATE,

    "validation_size":
        VALIDATION_SIZE,

    "training_observations":
        int(
            len(
                crime_training_features_split
            )
        ),

    "validation_observations":
        int(
            len(
                crime_validation_features_split
            )
        ),

    "feature_count":
        int(
            len(
                model_feature_columns
            )
        ),

    "numerical_feature_count":
        int(
            len(
                numerical_feature_columns
            )
        ),

    "categorical_feature_count":
        int(
            len(
                categorical_feature_columns
            )
        ),

    "model_features":
        model_feature_columns,

    "numerical_features":
        numerical_feature_columns,

    "categorical_features":
        categorical_feature_columns,

    "accuracy":
        float(
            crime_svm_accuracy
        ),

    "macro_f1":
        float(
            crime_svm_macro_f1
        ),

    "weighted_f1":
        float(
            crime_svm_weighted_f1
        )
}


with open(
    RESULTS_DIR
    / "crime_svm_model_metadata.json",
    "w",
    encoding="utf-8"
) as svm_metadata_file:

    json.dump(
        crime_svm_metadata,
        svm_metadata_file,
        indent=4
    )


# ============================================================
# Prepare Kaggle submission
# ============================================================

print(
    "\nGenerating SVM Kaggle submission..."
)

crime_svm_test_predictions = (
    crime_svm_classifier.predict(
        crime_test_model_features
    )
)


# ------------------------------------------------------------
# Create submission probability matrix
#
# LinearSVC does not produce calibrated probabilities.
# Therefore, the required Kaggle probability columns are
# constructed as a one-hot prediction matrix.
#
# This preserves the required submission structure while
# explicitly documenting that these are hard-class scores,
# not calibrated class probabilities.
# ------------------------------------------------------------

crime_submission_class_labels = (
    crime_sample_submission.columns[
        1:
    ]
)


crime_svm_submission = (
    crime_sample_submission.copy()
)


crime_svm_submission[
    crime_submission_class_labels
] = 0.0


crime_submission_class_to_column = {
    class_label: class_label
    for class_label
    in crime_submission_class_labels
}


for row_index, predicted_class in enumerate(
    crime_svm_test_predictions
):

    if predicted_class in (
        crime_submission_class_to_column
    ):

        crime_svm_submission.loc[
            row_index,
            predicted_class
        ] = 1.0


crime_svm_submission_path = (
    OUTPUT_DIR
    / "submissions"
)

crime_svm_submission_path.mkdir(
    parents=True,
    exist_ok=True
)


crime_svm_submission_file = (
    crime_svm_submission_path
    / "crime_svm_submission.csv"
)


crime_svm_submission.to_csv(
    crime_svm_submission_file,
    index=False
)


print(
    "Saved submission:",
    crime_svm_submission_file
)


# ============================================================
# Submission validation
# ============================================================

submission_expected_columns = (
    crime_sample_submission.columns.tolist()
)

submission_actual_columns = (
    crime_svm_submission.columns.tolist()
)


if (
    submission_actual_columns
    != submission_expected_columns
):

    raise ValueError(
        "SVM submission columns do not match "
        "the Kaggle sample submission."
    )


if len(
    crime_svm_submission
) != len(
    crime_sample_submission
):

    raise ValueError(
        "SVM submission row count does not match "
        "the Kaggle sample submission."
    )


if (
    crime_svm_submission["Id"].tolist()
    != crime_sample_submission["Id"].tolist()
):

    raise ValueError(
        "SVM submission Id values do not match "
        "the Kaggle sample submission."
    )


# ============================================================
# Durable submission metadata
# ============================================================

crime_svm_submission_metadata = {

    "submission_file":
        str(
            crime_svm_submission_file
        ),

    "rows":
        int(
            len(
                crime_svm_submission
            )
        ),

    "columns":
        int(
            len(
                crime_svm_submission.columns
            )
        ),

    "probability_columns":
        crime_submission_class_labels.tolist(),

    "prediction_type":
        "hard_class_one_hot",

    "probability_calibration":
        "LinearSVC does not provide calibrated probabilities.",

    "validation_accuracy":
        float(
            crime_svm_accuracy
        ),

    "validation_macro_f1":
        float(
            crime_svm_macro_f1
        ),

    "validation_weighted_f1":
        float(
            crime_svm_weighted_f1
        )
}


with open(
    RESULTS_DIR
    / "crime_svm_submission_metadata.json",
    "w",
    encoding="utf-8"
) as submission_metadata_file:

    json.dump(
        crime_svm_submission_metadata,
        submission_metadata_file,
        indent=4
    )


# ============================================================
# Final status
# ============================================================

print("\n" + "=" * 60)

print(
    "08_crime_svm.py completed successfully."
)

print("=" * 60)

print(
    "\nSVM validation accuracy:",
    crime_svm_accuracy
)

print(
    "SVM validation macro F1:",
    crime_svm_macro_f1
)

print(
    "SVM validation weighted F1:",
    crime_svm_weighted_f1
)

print(
    "\nSVM submission:",
    crime_svm_submission_file
)


print("\n" + "=" * 60)
print("08_crime_svm.py completed successfully")
print("=" * 60)

print("\n" + "=" * 60)
print(
    "\nAll SVM artifacts are ready for interpretation "
    "in SoarDDDS8555-8-Report.Rmd."
)

print("=" * 60)

