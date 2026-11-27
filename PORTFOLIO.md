# Portfolio and Interview Preparation

## Professional Project Description

Built an end-to-end credit-card fraud detection system using Python and Streamlit. The project performs EDA, preprocessing, stratified train-test evaluation, baseline and optimized model comparison, decision-threshold analysis, prediction risk scoring, and interactive dashboard visualization. The final class-weighted Logistic Regression model achieved 80% recall, 82.61% precision, 0.812834 F1-score, and 0.774204 PR-AUC on the held-out test set.

## Resume Bullets

- Built an end-to-end fraud-detection workflow using Pandas, NumPy, and Streamlit, covering EDA, preprocessing, model comparison, threshold tuning, prediction, and dashboard visualization.
- Prevented data leakage by fitting `Amount` and `Time` preprocessing only on training data and preserving a stratified 56,746-row held-out test set.
- Compared Logistic Regression, Decision Tree, Random Forest, class-weighted logistic, and SMOTE approaches using precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrices.
- Improved fraud recall from 53.68% in the baseline logistic model to 80% in the selected class-weighted model while limiting false positives to 16 on the test set.
- Developed reusable prediction validation, model loading, risk scoring, unit tests, deployment-safe data bundles, and Streamlit dashboard components.

## Technologies You Can Honestly Claim

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- pytest
- GitHub workflow preparation
- Custom NumPy/Pandas machine-learning implementations
- Streamlit Community Cloud deployment preparation

Do not claim native scikit-learn model training or a deployed public application because scikit-learn was excluded from the final runtime and no deployment has occurred.

## Interview Questions and Suggested Answers

### Why is the dataset imbalanced?

Fraud is rare: 492 fraud cases represent 0.1727% of 284,807 transactions. Accuracy alone is misleading because predicting every transaction as legitimate would still exceed 99.8% accuracy.

### How did you split the data?

I duplicated and cleaned the data in memory, separated `Class` from features, and created an 80/20 stratified split with seed 42. Stratification preserves the fraud ratio in both sets.

### How did you prevent data leakage?

I removed duplicates before splitting. I fit `Amount`/`Time` standardization only on training data and applied the fitted scaler to test and prediction inputs. SMOTE and threshold tuning were restricted to training-derived data.

### Which models did you test?

I tested Logistic Regression, Decision Tree, and Random Forest baselines, plus class-weighted Logistic Regression and SMOTE Logistic Regression.

### Why did you select class-weighted Logistic Regression?

It had the best F1-score and PR-AUC among the Phase 5 experiment options, while detecting 76 fraud cases with 16 false positives. SMOTE detected 84 fraud cases but caused 204 false positives.

### Why not choose the model with the highest accuracy?

Because accuracy is dominated by legitimate transactions. Precision, recall, F1-score, PR-AUC, false positives, and false negatives better measure fraud detection performance.

### What do precision and recall mean?

Precision is the fraction of fraud alerts that are truly fraud. Recall is the fraction of actual fraud cases detected.

### What is PR-AUC?

PR-AUC summarizes precision and recall across thresholds. It is especially informative for rare-positive datasets such as fraud detection.

### What is ROC-AUC?

ROC-AUC summarizes true-positive versus false-positive ranking across thresholds. It is useful but can be less directly informative than PR-AUC for highly imbalanced data.

### Why did you use threshold 0.40?

The threshold was selected from validation data by maximum F1-score, with ties favoring recall and fewer false positives. It was not assumed from the 0.50 default. The final test set was evaluated only afterward.

### What is a false positive and false negative?

A false positive is a legitimate transaction incorrectly flagged as fraud. A false negative is a fraudulent transaction missed by the model.

### How does class weighting differ from SMOTE?

Class weighting changes the training loss so fraud errors receive a larger penalty. SMOTE creates synthetic fraud samples. In this project, class weighting produced a better precision/recall balance; SMOTE substantially increased false positives.

### What are the project limitations?

It uses one public dataset, a random rather than temporal split, custom implementations due to environment restrictions, and a 0.40 threshold selected for one experiment. It is not a production monitoring or real-time investigation system.

### How is the dashboard deployed or published?

It is not publicly deployed. The local Streamlit dashboard is verified, deployment files are prepared, and Streamlit Community Cloud can use `app.py` as the main file after explicit GitHub publishing approval.
