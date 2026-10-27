"""End-to-end UCI Adult Income classification pipeline.

Downloads the official train/test files, cleans their values, fits a
preprocessing-and-classification pipeline, evaluates it, and writes the
metrics and a short conclusion to evaluation_results.txt.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
FEATURE_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
]
TARGET_COLUMN = "income"
NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
CATEGORICAL_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS
]
TARGET_NAMES = ["<=50K", ">50K"]


def download_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and parse UCI's official training and test files."""
    request = Request(DATASET_URL, headers={"User-Agent": "uci-adult-ml-assignment/1.0"})
    with urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = {Path(name).name.lower(): name for name in archive.namelist()}
        missing_members = {"adult.data", "adult.test"} - members.keys()
        if missing_members:
            raise ValueError(f"UCI archive is missing expected files: {sorted(missing_members)}")

        def read_split(member_name: str) -> pd.DataFrame:
            with archive.open(members[member_name]) as data_file:
                return pd.read_csv(
                    data_file,
                    header=None,
                    names=FEATURE_COLUMNS + [TARGET_COLUMN],
                    skipinitialspace=True,
                    comment="|",
                    na_values=["?"],
                    skip_blank_lines=True,
                )

        train = read_split("adult.data")
        test = read_split("adult.test")
    return train, test


def clean_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int, int]:
    """Normalize strings/labels and convert malformed numeric values to missing."""
    data = data.copy()
    if data.shape[1] != len(FEATURE_COLUMNS) + 1:
        raise ValueError(f"Expected 15 columns, received {data.shape[1]}")

    for column in CATEGORICAL_COLUMNS + [TARGET_COLUMN]:
        data[column] = data[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        data[column] = data[column].replace({"?": np.nan, "": np.nan})

    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN].astype("string").str.strip().str.rstrip(".").str.strip()
    )
    target = data[TARGET_COLUMN].map({"<=50K": 0, ">50K": 1})
    valid_target = target.notna()
    dropped_rows = int((~valid_target).sum())
    data = data.loc[valid_target].copy()
    target = target.loc[valid_target].astype("int64")

    features = data[FEATURE_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    missing_before_imputation = int(features.isna().sum().sum())
    return features, target, dropped_rows, missing_before_imputation


def main() -> None:
    train_raw, test_raw = download_dataset()
    X_train, y_train, dropped_train, missing_train = clean_split(train_raw)
    X_test, y_test, dropped_test, missing_test = clean_split(test_raw)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    positive_probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, positive_probabilities)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    conclusion = (
        f"On {len(y_test):,} official UCI test examples, the logistic-regression "
        f"pipeline achieved an F1-score of {f1:.3f} and ROC-AUC of {roc_auc:.3f} "
        f"for the >50K class. The F1-score summarizes the precision/recall balance; "
        f"ROC-AUC measures ranking quality across classification thresholds. "
        f"Because class_weight='balanced' prioritizes the minority class, review "
        f"precision and recall together when deciding whether this model suits "
        f"the intended use."
    )
    report = f"""UCI Adult Income Classification - Evaluation Results
========================================================
Dataset: UCI Machine Learning Repository, Adult (Census Income)
Dataset page: https://archive.ics.uci.edu/dataset/2/adult
Split: official UCI training and test files
Target: income (>50K is the positive class)

Data cleaning:
- Trimmed whitespace and normalized the test labels' trailing period.
- Treated '?' and empty strings as missing values.
- Converted numeric columns to numbers; unparseable values become missing.
- Dropped rows with labels other than <=50K or >50K.
- Training rows: {len(y_train):,} (dropped invalid labels: {dropped_train})
- Test rows: {len(y_test):,} (dropped invalid labels: {dropped_test})
- Missing feature values before imputation: train={missing_train:,}, test={missing_test:,}

Preprocessing and model:
- Numeric: median imputation, then standard scaling.
- Categorical: most-frequent imputation, then one-hot encoding.
- Unknown test categories are ignored by the encoder.
- Model: class-weight-balanced logistic regression (max_iter=1000).

Evaluation metrics (positive class: >50K):
- Accuracy:  {accuracy:.4f} - fraction of all test predictions that are correct.
- Precision: {precision:.4f} - among predicted >50K cases, the fraction truly >50K.
- Recall:    {recall:.4f} - among actual >50K cases, the fraction found by the model.
- F1-score:  {f1:.4f} - harmonic mean of precision and recall; useful when balancing them.
- ROC-AUC:   {roc_auc:.4f} - ability to rank positive cases above negative cases across thresholds;
  0.5 is chance-level and 1.0 is perfect ranking.

Confusion matrix (rows=actual, columns=predicted; order: <=50K, >50K):
                 predicted <=50K   predicted >50K
actual <=50K          {tn:>8,}          {fp:>8,}
actual >50K            {fn:>8,}          {tp:>8,}
True negatives={tn:,}; false positives={fp:,}; false negatives={fn:,}; true positives={tp:,}.

Per-class precision, recall, and F1:
{classification_report(y_test, predictions, target_names=TARGET_NAMES, digits=4, zero_division=0)}
Conclusion:
{conclusion}
"""

    output_path = Path(__file__).with_name("evaluation_results.txt")
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
