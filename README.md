# Assignment 8 — Predictive Modeling and Analytics

## Overview

Assignment 8 examines the development, evaluation, and interpretation of predictive models across three major analytical tasks:

1. **Store Sales Time Series Forecasting**
2. **House Prices Advanced Regression**
3. **San Francisco Crime Classification**

The assignment applies statistical forecasting, regression, and machine-learning classification techniques together with quantitative model validation, model-selection procedures, assumption assessment, feature engineering, dimension reduction, and competition prediction.

The computational analysis was performed primarily in **Python**. Thirteen Python scripts were developed to perform data inspection, exploratory analysis, model development, validation, forecasting, classification, regression, feature engineering, dimension reduction, and generation of final prediction artifacts.

The resulting analysis is interpreted through an **R Markdown report**. The report reads durable analysis artifacts rather than rerunning the computationally intensive Python models. This separation improves reproducibility and avoids unnecessary model refitting during report generation.

---

## Analytical Components

### 1. Store Sales Time Series Forecasting

The store-sales component investigates time-dependent sales patterns and develops forecasting models using:

- Exponential smoothing / ETS
- ARIMA
- Time-series exploratory analysis
- Validation-based model comparison
- Forecast generation
- Competition submission preparation and validation

The analysis uses validation observations for model evaluation while keeping final competition test observations separate from model-selection decisions.

Relevant scripts:

02timeserieseda.py 03storesalesmodels.py 04storesalesets.py 05storesalesarima.py 11storesalesarimasubmission.py 12storesalesetssubmission.py 13storesalessubmissionvalidation.py


---

### 2. House Prices Advanced Regression

The House Prices component develops regression models using engineered and transformed predictors.

The analysis includes techniques such as:

- Feature engineering
- Variable transformations
- Interaction terms
- Polynomial terms
- Categorical encoding
- Subset selection
- Principal Component Analysis (PCA)
- Regression model comparison
- Validation-based performance evaluation

Relevant scripts:

09housepricesmodels.py 10housepricesrequired_techniques.py


---

### 3. San Francisco Crime Classification

The crime-classification component develops and evaluates several supervised classification approaches:

- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

The models are evaluated using validation data, with model-selection decisions made before generating predictions for the competition test observations.

Relevant scripts:

06crimemodels.py 07crimerequiredmodels.py 08crime_svm.py


---

## Project Structure

SoarDDDS8555-8/ ├── .gitignore ├── README.md ├── SoarDDDS8555-8.Rproj ├── SoarDDDS8555-8-Main-Controller.Rmd ├── SoarDDDS8555-8-Report.Rmd ├── SoarDDDS8555-8-Report.pdf ├── SoarDDDS8555-8-Report.html │ ├── data/ │ ├── housepricesadvancedregressiontechniques/ │ ├── sanfranciscocrimeclassification/ │ └── storesalestimeseriesforecasting/ │ ├── scripts/ │ ├── 01datainspection.py │ ├── 02timeserieseda.py │ ├── 03storesalesmodels.py │ ├── 04storesalesets.py │ ├── 05storesalesarima.py │ ├── 06crimemodels.py │ ├── 07crimerequiredmodels.py │ ├── 08crimesvm.py │ ├── 09housepricesmodels.py │ ├── 10housepricesrequiredtechniques.py │ ├── 11storesalesarimasubmission.py │ ├── 12storesalesetssubmission.py │ └── 13storesalessubmission_validation.py │ └── output/ └── Generated locally; not tracked by Git


---

## Computational Workflow

The analysis follows a staged workflow:

1. Inspect and understand the input datasets.
2. Perform exploratory data analysis.
3. Engineer and transform predictors where appropriate.
4. Develop candidate statistical and machine-learning models.
5. Evaluate models using validation data.
6. Compare candidate models using appropriate performance measures.
7. Investigate relevant assumptions and limitations.
8. Select final modeling approaches.
9. Generate competition predictions using the final models.
10. Validate the generated prediction artifacts.
11. Interpret the computational results through the R Markdown report.

---

## Validation Strategy

Validation data are used for model evaluation and model-selection decisions.

The final competition test observations are kept separate from validation. Because competition test datasets do not provide observed target values for ordinary local validation, test observations are used only after model development and selection.

This separation helps reduce the risk of using test information during model development.

---

## Reproducibility

The project separates computational model fitting from report interpretation.

The Python scripts perform the computationally intensive analysis and generate durable artifacts. The R Markdown report reads and interprets these artifacts rather than refitting the complete collection of models during the knitting process.

This approach:

- Reduces unnecessary computation during report generation.
- Makes the report easier to reproduce.
- Separates model development from result interpretation.
- Preserves the outputs needed to document the completed analysis.

The `output/` directory contains generated analysis artifacts and is intentionally excluded from version control because it can be regenerated by the analysis scripts.

---

## Data

The datasets used in this project are publicly available competition datasets associated with Kaggle competitions:

- House Prices: Advanced Regression Techniques
- San Francisco Crime Classification
- Store Sales Time Series Forecasting

Supporting competition files are included in the `data/` directory where appropriate.

### Large Training Dataset Note

Two original Kaggle training files exceed GitHub's standard 100 MiB individual-file limit:

data/sanfranciscocrimeclassification/train.csv data/storesalestimeseries_forecasting/train.csv


These files are therefore intentionally excluded from the Git repository.

The files remain available in the local project directory for computational analysis and are listed in `.gitignore` to prevent accidental commits.

The remaining competition data files are included where their individual file sizes permit.

---

## Output Files

Generated analysis artifacts are stored locally in:

output/


The output directory is intentionally not tracked by Git because its contents can be regenerated by running the analysis scripts.

The final interpreted reports are tracked separately:

SoarDDDS8555-8-Report.pdf SoarDDDS8555-8-Report.html SoarDDDS8555-8-Report.Rmd


---

## Technologies

The project uses:

- **Python** for computational analysis and predictive modeling
- **R / R Markdown** for interpretation and report generation
- **Git / GitHub** for version control
- **Kaggle competition datasets** for the predictive modeling tasks

The Python analysis includes statistical forecasting, regression, classification, feature engineering, model evaluation, and prediction generation.

---

## Author

**Soar**

Assignment 8 — DDS 8555

