# ============================================================
# Assignment 8: Build & Evaluate
# Time Series, Regression, and Classification Models
#
# 07_crime_required_models.py
#
# Purpose:
#   Build and evaluate the classification models required
#   for the San Francisco Crime Classification competition.
#
# Focus:
#   - Reproducible crime classification data preparation
#   - Train/validation model evaluation
#   - Decision tree classification
#   - Random forest classification
#   - Support Vector Machine classification
#   - Model assumptions and diagnostics
#   - Kaggle-ready prediction artifacts
#   - Report-ready evaluation artifacts
#
# Model requirements:
#   The assignment requires:
#   - Two tree-based classification models
#   - One Support Vector Machine
#
# Report workflow:
#   This script creates durable tables, figures, prediction
#   files, and JSON results for interpretation and presentation
#   in SoarDDDS8555-8-Report.Rmd.
#
# Design note:
#   This script is intentionally separate from
#   04_crime_models.py so that the required classification
#   models can be developed without rerunning the earlier
#   exploratory and baseline modeling workflow.
# ============================================================


from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss
  )

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings(
  "ignore",
  category=FutureWarning
)

# =========================================================
# PROJECT PATHS
# =========================================================

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

SUBMISSIONS_DIR = (
  OUTPUT_DIR
  / "submissions"
)

for output_directory in [
  TABLES_DIR,
  FIGURES_DIR,
  RESULTS_DIR,
  SUBMISSIONS_DIR
]:

  output_directory.mkdir(
    parents=True,
    exist_ok=True
  )

# =========================================================
# MODEL CONFIGURATION
# =========================================================

RANDOM_STATE = 8555

VALIDATION_SIZE = 0.20

DECISION_TREE_MAX_DEPTH = 18

DECISION_TREE_MIN_SAMPLES_LEAF = 20

RANDOM_FOREST_ESTIMATORS = 150

RANDOM_FOREST_MAX_DEPTH = 18

RANDOM_FOREST_MIN_SAMPLES_LEAF = 10

RANDOM_FOREST_MAX_FEATURES = "sqrt"

RANDOM_FOREST_N_JOBS = -1

# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("07_crime_required_models.py")
print("=" * 60)

print("\nLoading San Francisco Crime Classification data...")

crime_training_data = pd.read_csv(
  CRIME_DIR / "train.csv"
)

crime_test_data = pd.read_csv(
  CRIME_DIR / "test.csv"
)

crime_sample_submission = pd.read_csv(
  CRIME_DIR / "sample-submission.csv"
)

print(
  "Training data shape:",
  crime_training_data.shape
)

print(
  "Test data shape:",
  crime_test_data.shape
)

print(
  "Sample submission shape:",
  crime_sample_submission.shape
)

# =========================================================
# TARGET AND DATE FEATURE ENGINEERING
# =========================================================

crime_target_column = "Category"

crime_training_features = (
  crime_training_data.copy()
)

crime_test_features = (
  crime_test_data.copy()
)

crime_training_features["Dates"] = pd.to_datetime(
  crime_training_features["Dates"]
)

crime_test_features["Dates"] = pd.to_datetime(
  crime_test_features["Dates"]
)

for data_frame in [
  crime_training_features,
  crime_test_features
]:

  data_frame["date_year"] = (
    data_frame["Dates"].dt.year
  )

  data_frame["date_month"] = (
    data_frame["Dates"].dt.month
  )

  data_frame["date_day"] = (
    data_frame["Dates"].dt.day
  )

  data_frame["date_hour"] = (
    data_frame["Dates"].dt.hour
  )

  data_frame["date_day_of_week"] = (
    data_frame["Dates"].dt.dayofweek
  )

# =========================================================
# MODEL FEATURE DEFINITION
# =========================================================
# Descript and Resolution are excluded because they are not
# available in the competition test data.
# Dates itself is excluded after extracting useful temporal
# components.
#
# Address is excluded here to keep the required models
# computationally manageable and to maintain consistency
# with the completed 04_ modeling analysis.
# =========================================================

model_feature_columns = [
  "DayOfWeek",
  "PdDistrict",
  "X",
  "Y",
  "date_year",
  "date_month",
  "date_day",
  "date_hour",
  "date_day_of_week"
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

numerical_feature_columns = (
  crime_model_features
  .select_dtypes(
    include=["number", "bool"]
  )
  .columns
  .tolist()
)

categorical_feature_columns = (
  crime_model_features
  .select_dtypes(
    include=["object", "category"]
  )
  .columns
  .tolist()
)

print("\nModel feature columns:")
print(model_feature_columns)

print(
  "\nNumerical feature count:",
  len(numerical_feature_columns)
)

print(
  "Categorical feature count:",
  len(categorical_feature_columns)
)

# =========================================================
# REPRODUCIBLE VALIDATION SPLIT
# =========================================================

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

print("\nClassification validation split")

print(
  "Training observations:",
  len(crime_training_features_split)
)

print(
  "Validation observations:",
  len(crime_validation_features_split)
)

crime_validation_split_summary = pd.DataFrame({
  
  "dataset": [
      "model_training",
      "validation"
  ],
  
  "observations": [
      len(crime_training_features_split),
      len(crime_validation_features_split)
  ],
  
  "percentage": [
      (
          len(crime_training_features_split)
          / len(crime_model_features)
          * 100
      ),
  
      (
          len(crime_validation_features_split)
          / len(crime_model_features)
          * 100
      )
  ]
  

})

crime_validation_split_summary.to_csv(
  TABLES_DIR
  / "crime_required_models_validation_split_summary.csv",
  index=False
)

# =========================================================
# PREPROCESSING
# =========================================================

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
# DECISION TREE
# =========================================================

print("\nFitting Decision Tree...")

crime_decision_tree_classifier = Pipeline(
  steps=[
    (
      "preprocessor",
      crime_preprocessor
    ),
    (
      "classifier",
        DecisionTreeClassifier(
          max_depth=DECISION_TREE_MAX_DEPTH,
          min_samples_leaf=(
            DECISION_TREE_MIN_SAMPLES_LEAF
          ),
          random_state=RANDOM_STATE
        )
    )
  ]
)

crime_decision_tree_classifier.fit(
  crime_training_features_split,
  crime_training_target_split
)

crime_decision_tree_predictions = (
  crime_decision_tree_classifier.predict(
    crime_validation_features_split
  )
)

crime_decision_tree_probabilities = (
  crime_decision_tree_classifier.predict_proba(
    crime_validation_features_split
  )
)

# =========================================================
# RANDOM FOREST
# =========================================================

print("\nFitting Random Forest...")

crime_random_forest_classifier = Pipeline(
  steps=[
    (
      "preprocessor",
      crime_preprocessor
    ),
    (
      "classifier",
      RandomForestClassifier(
        n_estimators=(
          RANDOM_FOREST_ESTIMATORS
        ),
        max_depth=(
          RANDOM_FOREST_MAX_DEPTH
        ),
        min_samples_leaf=(
          RANDOM_FOREST_MIN_SAMPLES_LEAF
        ),
        max_features=(
          RANDOM_FOREST_MAX_FEATURES
        ),
        n_jobs=RANDOM_FOREST_N_JOBS,
        random_state=RANDOM_STATE
      )
    )
  ]
)

crime_random_forest_classifier.fit(
  crime_training_features_split,
  crime_training_target_split
)

crime_random_forest_predictions = (
crime_random_forest_classifier.predict(
crime_validation_features_split
)
)

crime_random_forest_probabilities = (
crime_random_forest_classifier.predict_proba(
crime_validation_features_split
)
)

# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_crime_classifier(
    model_name,
    class_labels,
    model_predictions,
    model_probabilities,
    validation_target
):

    model_accuracy = accuracy_score(
        validation_target,
        model_predictions
    )

    model_log_loss = log_loss(
        validation_target,
        model_probabilities,
        labels=class_labels
    )

    model_classification_report = (
        classification_report(
            validation_target,
            model_predictions,
            output_dict=True,
            zero_division=0
        )
    )

    model_macro_f1 = (
        model_classification_report[
            "macro avg"
        ][
            "f1-score"
        ]
    )

    model_weighted_f1 = (
        model_classification_report[
            "weighted avg"
        ][
            "f1-score"
        ]
    )

    return {
        "model": model_name,
        "accuracy": model_accuracy,
        "log_loss": model_log_loss,
        "macro_f1": model_macro_f1,
        "weighted_f1": model_weighted_f1
    }


crime_decision_tree_classes = (
    crime_decision_tree_classifier[
        "classifier"
    ].classes_
)


crime_random_forest_classes = (
    crime_random_forest_classifier[
        "classifier"
    ].classes_
)


crime_decision_tree_metrics = (
    evaluate_crime_classifier(
        "Decision Tree",
        crime_decision_tree_classes,
        crime_decision_tree_predictions,
        crime_decision_tree_probabilities,
        crime_validation_target_split
    )
)


crime_random_forest_metrics = (
    evaluate_crime_classifier(
        "Random Forest",
        crime_random_forest_classes,
        crime_random_forest_predictions,
        crime_random_forest_probabilities,
        crime_validation_target_split
    )
)


crime_tree_model_evaluation = pd.DataFrame([
    crime_decision_tree_metrics,
    crime_random_forest_metrics
])


print(
    "\nTree model evaluation:"
)

print(
    crime_tree_model_evaluation.to_string(
        index=False
    )
)


crime_tree_model_evaluation.to_csv(
    TABLES_DIR
    / "crime_tree_model_evaluation.csv",
    index=False
)


# =========================================================
# CLASSIFICATION REPORTS
# =========================================================


crime_decision_tree_report = pd.DataFrame(
classification_report(
crime_validation_target_split,
crime_decision_tree_predictions,
output_dict=True,
zero_division=0
)
).transpose()

crime_decision_tree_report.to_csv(
TABLES_DIR
/ "crime_decision_tree_classification_report.csv"
)

crime_random_forest_report = pd.DataFrame(
classification_report(
crime_validation_target_split,
crime_random_forest_predictions,
output_dict=True,
zero_division=0
)
).transpose()

crime_random_forest_report.to_csv(
TABLES_DIR
/ "crime_random_forest_classification_report.csv"
)

# =========================================================
# CONFUSION MATRICES
# =========================================================

crime_decision_tree_confusion_matrix = (
confusion_matrix(
crime_validation_target_split,
crime_decision_tree_predictions,
labels=crime_decision_tree_classes
)
)

crime_random_forest_confusion_matrix = (
confusion_matrix(
crime_validation_target_split,
crime_random_forest_predictions,
labels=crime_random_forest_classes
)
)

pd.DataFrame(
crime_decision_tree_confusion_matrix,
index=crime_decision_tree_classes,
columns=crime_decision_tree_classes
).to_csv(
TABLES_DIR
/ "crime_decision_tree_confusion_matrix.csv"
)

pd.DataFrame(
crime_random_forest_confusion_matrix,
index=crime_random_forest_classes,
columns=crime_random_forest_classes
).to_csv(
TABLES_DIR
/ "crime_random_forest_confusion_matrix.csv"
)

# =========================================================
# MODEL COMPARISON FIGURE
# =========================================================

crime_tree_model_plot_data = (
crime_tree_model_evaluation
.melt(
id_vars="model",
value_vars=[
"accuracy",
"macro_f1",
"weighted_f1"
],
var_name="metric",
value_name="value"
)
)

plt.figure(
figsize=(10, 6)
)

sns.barplot(
data=crime_tree_model_plot_data,
x="metric",
y="value",
hue="model"
)

plt.title(
"Required Tree Model Validation Performance"
)

plt.xlabel("Metric")

plt.ylabel("Score")

plt.ylim(0, 1)

plt.grid(
axis="y",
alpha=0.3
)

plt.tight_layout()

plt.savefig(
FIGURES_DIR
/ "crime_tree_model_comparison.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# =========================================================
# DURABLE MODEL METADATA
# =========================================================

crime_required_model_metadata = {

"target_column":
    crime_target_column,

"training_rows":
    int(len(crime_training_data)),

"test_rows":
    int(len(crime_test_data)),

"validation_rows":
    int(len(crime_validation_features_split)),

"random_state":
    RANDOM_STATE,

"validation_size":
    VALIDATION_SIZE,

"model_features":
    model_feature_columns,

"numerical_features":
    numerical_feature_columns,

"categorical_features":
    categorical_feature_columns,

"excluded_columns": [
    "Dates",
    "Descript",
    "Resolution",
    "Id",
    "Address"
],

"decision_tree_parameters": {

    "max_depth":
        DECISION_TREE_MAX_DEPTH,

    "min_samples_leaf":
        DECISION_TREE_MIN_SAMPLES_LEAF
},

"random_forest_parameters": {

    "n_estimators":
        RANDOM_FOREST_ESTIMATORS,

    "max_depth":
        RANDOM_FOREST_MAX_DEPTH,

    "min_samples_leaf":
        RANDOM_FOREST_MIN_SAMPLES_LEAF,

    "max_features":
        RANDOM_FOREST_MAX_FEATURES,

    "n_jobs":
        RANDOM_FOREST_N_JOBS
}


}

with open(
RESULTS_DIR
/ "crime_required_model_metadata.json",
"w",
encoding="utf-8"
) as metadata_file:

  json.dump(
      crime_required_model_metadata,
      metadata_file,
      indent=4
  )

# =========================================================
# SECTION COMPLETE
# =========================================================

print("\nTree model analysis completed.")

print(
"Saved tree evaluation, classification reports, "
"confusion matrices, comparison figure, and metadata."
)

print(
"Ready for the SVM and submission section."
)

# =========================================================
# KAGGLE SUBMISSION FILES
# =========================================================
#
# Final Kaggle models are refit using the complete
# competition training dataset before predicting the
# competition test dataset.
#
# Kaggle requires one probability column for each
# crime category, plus Id.
# =========================================================

print("\n" + "=" * 60)
print("Generating Kaggle Crime Classification submissions")
print("=" * 60)


# ---------------------------------------------------------
# Rebuild final Decision Tree using ALL training data
# ---------------------------------------------------------

print("\nFitting final Decision Tree on all training data...")

crime_decision_tree_final = Pipeline(
    steps=[
        (
            "preprocessor",
            crime_preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=DECISION_TREE_MAX_DEPTH,
                min_samples_leaf=(
                    DECISION_TREE_MIN_SAMPLES_LEAF
                ),
                random_state=RANDOM_STATE
            )
        )
    ]
)

crime_decision_tree_final.fit(
    crime_model_features,
    crime_model_target
)


# ---------------------------------------------------------
# Decision Tree test probabilities
# ---------------------------------------------------------

crime_decision_tree_test_probabilities = (
    crime_decision_tree_final.predict_proba(
        crime_test_model_features
    )
)

crime_decision_tree_test_classes = (
    crime_decision_tree_final[
        "classifier"
    ].classes_
)


# ---------------------------------------------------------
# Create Decision Tree submission
# ---------------------------------------------------------

crime_decision_tree_submission = pd.DataFrame(
    crime_decision_tree_test_probabilities,
    columns=crime_decision_tree_test_classes
)

crime_decision_tree_submission.insert(
    0,
    "Id",
    crime_test_data["Id"].values
)


# Ensure exact sample-submission column order
crime_decision_tree_submission = (
    crime_decision_tree_submission[
        crime_sample_submission.columns
    ]
)


crime_decision_tree_submission_path = (
    SUBMISSIONS_DIR
    / "crime_decision_tree_submission.csv"
)

crime_decision_tree_submission.to_csv(
    crime_decision_tree_submission_path,
    index=False
)


print(
    "\nDecision Tree Kaggle submission saved:"
)

print(
    crime_decision_tree_submission_path
)

print(
    "Shape:",
    crime_decision_tree_submission.shape
)


# ---------------------------------------------------------
# Rebuild final Random Forest using ALL training data
# ---------------------------------------------------------

print(
    "\nFitting final Random Forest on all training data..."
)

crime_random_forest_final = Pipeline(
    steps=[
        (
            "preprocessor",
            crime_preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=RANDOM_FOREST_ESTIMATORS,
                max_depth=RANDOM_FOREST_MAX_DEPTH,
                min_samples_leaf=(
                    RANDOM_FOREST_MIN_SAMPLES_LEAF
                ),
                max_features=(
                    RANDOM_FOREST_MAX_FEATURES
                ),
                n_jobs=RANDOM_FOREST_N_JOBS,
                random_state=RANDOM_STATE
            )
        )
    ]
)

crime_random_forest_final.fit(
    crime_model_features,
    crime_model_target
)


# ---------------------------------------------------------
# Random Forest test probabilities
# ---------------------------------------------------------

crime_random_forest_test_probabilities = (
    crime_random_forest_final.predict_proba(
        crime_test_model_features
    )
)

crime_random_forest_test_classes = (
    crime_random_forest_final[
        "classifier"
    ].classes_
)


# ---------------------------------------------------------
# Create Random Forest submission
# ---------------------------------------------------------

crime_random_forest_submission = pd.DataFrame(
    crime_random_forest_test_probabilities,
    columns=crime_random_forest_test_classes
)

crime_random_forest_submission.insert(
    0,
    "Id",
    crime_test_data["Id"].values
)


# Ensure exact sample-submission column order
crime_random_forest_submission = (
    crime_random_forest_submission[
        crime_sample_submission.columns
    ]
)


crime_random_forest_submission_path = (
    SUBMISSIONS_DIR
    / "crime_random_forest_submission.csv"
)

crime_random_forest_submission.to_csv(
    crime_random_forest_submission_path,
    index=False
)


print(
    "\nRandom Forest Kaggle submission saved:"
)

print(
    crime_random_forest_submission_path
)

print(
    "Shape:",
    crime_random_forest_submission.shape
)


# =========================================================
# SUBMISSION VALIDATION
# =========================================================

print("\n" + "-" * 60)
print("Validating Kaggle submission files")
print("-" * 60)


for submission_path in [
    crime_decision_tree_submission_path,
    crime_random_forest_submission_path
]:

    submission_data = pd.read_csv(
        submission_path
    )

    print(
        f"\n{submission_path.name}"
    )

    print(
        "Rows:",
        len(submission_data)
    )

    print(
        "Columns:",
        submission_data.columns.tolist()
    )

    print(
        "Expected rows:",
        len(crime_test_data)
    )

    print(
        "Matches sample columns:",
        (
            submission_data.columns.tolist()
            ==
            crime_sample_submission.columns.tolist()
        )
    )

    print(
        "Missing values:",
        submission_data.isna().sum().sum()
    )


print("\n" + "=" * 60)
print("Crime Kaggle submission generation completed.")
print("=" * 60)
