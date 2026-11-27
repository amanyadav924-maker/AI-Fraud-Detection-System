# Testing Documentation

## Framework

The project uses `pytest`, executed with the virtual environment's Python interpreter. Install `requirements-dev.txt` to include the test and exploratory-analysis dependencies.

## Test Areas

### Preprocessing

- Valid dataset acceptance.
- Missing-value rejection.
- Invalid `Class` value rejection.
- Duplicate removal without modifying the source DataFrame.
- Feature/target separation.
- Reproducible stratified split with both classes.
- Training-only scaling behavior.

### Model Training

- Confusion-matrix layout.
- Perfect-classifier metrics.
- Zero-positive-prediction edge case.
- ROC-AUC perfect ranking.
- Average precision perfect ranking.
- Valid logistic, decision-tree, and random-forest probabilities/predictions.

### Imbalance and Optimization

- SMOTE-style training resampling and class balance.
- Training-derived stratified validation split.
- Weighted logistic probabilities.
- Validation threshold search.

### Final Selection

- Threshold result structure.
- Highest-F1 threshold selection.
- Threshold selection not hard-coded to 0.50.
- Empty-threshold error handling.

### Prediction Engine

- Mapping and sequence input acceptance.
- Missing feature rejection.
- Non-numeric feature rejection.
- Incorrect feature count rejection.
- Low, medium, and high risk categories.
- Fraud and legitimate predictions.
- Missing model-file handling.
- Invalid model artifact handling.

### Dashboard

- Dataset summary calculations.
- Class and amount filtering.
- Actual final-report parsing.
- Cloud deployment summary loading.
- Deployment test-row loading.
- Streamlit `AppTest` loading with no exceptions.

## Actual Executed Results

Latest local verification:

```text
39 passed
```

The suite was executed with:

```powershell
python -m pytest -q
```

The runtime requirements were installed and checked on Linux Python 3.12.3. A Linux cloud-mode smoke test loaded the final model and scaler, predicted from a deployment sample, loaded the summary and evaluation report without the raw dataset, and exercised all dashboard pages through Streamlit AppTest. The local Streamlit server also returned HTTP 200.

## Limitations

- Tests validate logic and formatting, not a real financial deployment.
- AppTest does not reproduce every browser interaction or visual layout.
- The full dashboard is not covered by automated browser screenshots.
