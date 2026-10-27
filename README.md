# UCI Adult Income: End-to-End Machine Learning Pipeline

Part of the AI-Tasks repository for AI-related assignments.

This assignment builds a binary classifier to predict whether an individual's
income exceeds $50,000 using the UCI Adult (Census Income) dataset.

## Dataset

The script downloads the official archive from the
[UCI Machine Learning Repository — Adult](https://archive.ics.uci.edu/dataset/2/adult)
at runtime. It uses UCI's supplied training and test splits. The data contains
numeric and categorical predictors, missing values represented by `?`, and
income labels in the test file with a trailing period.

## Run

From this directory in Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\sample.py
```

The script prints the evaluation and saves the same report to
`evaluation_results.txt` beside the source file. The report includes dataset
cleaning counts, all requested metrics, the confusion matrix, a per-class
breakdown, metric explanations, and a short conclusion.

## Run on Render

The repository includes a Render Blueprint in `render.yaml` for a Cron Job.
To connect it to your Render account:

1. Open the [Render Cron Job setup](https://dashboard.render.com/cron/new) and
   connect your GitHub account if prompted.
2. Select `Ushasri220/AI-Tasks` and the `main` branch. Render can use the
   Blueprint in the repository to configure the job.
3. After the first deployment, open the job's **Runs** page and choose
   **Trigger Run** to run the pipeline immediately.
4. The Blueprint schedule is set to January 1 yearly because Render Cron Jobs
   require a schedule. If this is for a one-time run, suspend or delete the
   Cron Job after the manual run completes to prevent future runs and charges.

The Cron Job downloads the dataset each time it runs and writes its evaluation
report to the job logs. Render Cron Jobs do not retain local files after a run,
so `evaluation_results.txt` is not a persistent downloadable artifact.

## Pipeline

- Trims inconsistent whitespace, treats `?` and empty strings as missing, and
  normalizes the test labels' trailing period.
- Converts numeric columns safely; invalid numeric values become missing.
  Rows with invalid target labels are excluded.
- Imputes numeric values with the median and scales them with `StandardScaler`.
- Imputes categorical values with the most frequent category and one-hot
  encodes them, ignoring categories not seen during training.
- Fits class-weight-balanced logistic regression on the official training
  split and evaluates once on the official test split.

## Metrics

- **Accuracy** is the fraction of all test predictions that are correct. It
  should be interpreted with the other metrics because the classes are
  imbalanced.
- **Precision** is the fraction of predicted `>50K` examples that truly have
  income `>50K`.
- **Recall** is the fraction of actual `>50K` examples the model identifies.
- **F1-score** is the harmonic mean of precision and recall and summarizes
  their balance.
- **ROC-AUC** measures how well the model ranks positive examples above
  negative examples over all classification thresholds; 0.5 is chance-level
  and 1.0 is perfect ranking.
- The **confusion matrix** shows true negatives, false positives, false
  negatives, and true positives for the fixed classification threshold.

This is a baseline for educational use, not a basis for consequential
decisions about individuals.
