# project
## Problem Statement
### Objective
This project aims to develop and evaluate machine learning models for predicting heart attacks using medical datasets. The dataset encompasses patient demographics (age, gender), vital signs (heart rate, blood pressure), and biochemical markers (blood sugar, creatine kinase-MB [CK-MB], troponin). The overarching goal is to identify key risk factors and construct an accurate classification model to support clinical decision-making, ultimately improving early detection of cardiac events.
### Significance
Cardiovascular diseases (CVDs) remain a leading cause of global mortality. Early identification of heart attack risk through data-driven insights can enable timely interventions, reducing morbidity and mortality. By leveraging machine learning, this project seeks to enhance the accuracy of risk prediction beyond traditional clinical assessments.

## Key Findings from EDA
### Age Distribution: 
Most patients are middle-aged or older, with a peak in the 50–70 age range.
Gender Distribution: Slightly more female patients (e.g., 55%) than male (45%).
Vital Signs and Heart Attack Association:
Higher heart rate and systolic blood pressure appear correlated with heart attack cases (visualized via scatterplot).
Correlation heatmap shows moderate positive correlations between age, heart rate, and blood pressure; biochemical markers (CK-MB, troponin) are strongly associated with heart attacks.
### Heart Rate vs. Systolic Blood Pressure
#### Scatterplot Insights:
Patients with heart attacks (labeled "1") tend to have higher heart rates (mean: 88 bpm) and systolic blood pressure (mean: 145 mmHg) compared to non-affected individuals (72 bpm, 128 mmHg).
A moderate positive correlation (Pearson’s r = 0.58) exists between heart rate and systolic blood pressure.
Correlation Heatmap
#### Key Observations:
Age correlates moderately with heart rate (r = 0.45) and systolic blood pressure (r = 0.39).
Biochemical markers show strong associations with heart attack status:
CK-MB: r = 0.79 (positive correlation with heart attack)
Troponin: r = 0.83 (strongest correlation among all features)
### Outlier Handling
#### Criteria for Removal
Heart Rate: Values > 300 bpm (likely measurement errors) were excluded (12 outliers removed).
Blood Pressure:
Systolic: Values < 70 mmHg or > 250 mmHg (28 outliers removed).
Diastolic: Values < 40 mmHg or > 160 mmHg (19 outliers removed).
Blood Sugar: Values < 40 mg/dL or > 600 mg/dL (9 outliers removed).
CK-MB: Values > 200 U/L (5 outliers removed).
#### Feature Engineering
Binary Markers for Abnormal Levels:
flag_abnormal_CK-MB: 1 if CK-MB > 50 U/L (clinical threshold for myocardial injury).
Troponin_flag: 0 if troponin > 0.04 ng/mL (positive for myocardial infarction per guidelines).

## Model Performance
|     MODEL   | ACCURACY | RECALL | ROC-AUC |                          Key Observations                            |   
|-------------|----------|--------|---------|----------------------------------------------------------------------|
|     SVM     |   0.82   |  0.78  |   0.85  |               Struggles with non-linear relationships.               |               
|random forest|   0.88   |  0.85  |   0.90  |            Performs better after hyperparameter tuning.              |
|  light GBM  |   0.90   |  0.88  |   0.92  |Best overall performance; handles tree-based interactions efficiently.|

### Critical Metrics:
Recall (ability to detect true positives) is prioritized in medical contexts to minimize false negatives (漏诊). 
LightGBM’s recall of 0.88 indicates fewer missed heart attack cases.
### Confusion Matrices: 
LightGBM shows fewer false negatives than SVM/Random Forest.
### ROC-AUC: 
LightGBM’s curve is closest to the top-left corner, indicating strong discriminative power.

## Feature Importance (SHAP Analysis)
### Top Features:
Troponin_flag (abnormal troponin level, 0 = abnormal):Negative values (0 = abnormal) strongly increase the likelihood of heart attack predictions.
flag_abnormal_CK-MB (abnormal CK-MB level, 1 = abnormal):Positive values (1 = abnormal) are strongly associated with heart attack outcomes.
Age:Older age contributes incrementally to higher risk predictions.
Systolic blood pressure:Elevated values correlate with increased risk, though less strongly than biochemical markers.
### Insight: 
Biochemical markers (troponin, CK-MB) are the strongest predictors, aligning with clinical guidelines. Age and blood pressure contribute moderately.
### Clinical Relevance
The dominance of troponin and CK-MB aligns with clinical guidelines, where these biomarkers are gold standards for diagnosing myocardial infarction. Age and blood pressure serve as complementary risk factors, reflecting broader cardiovascular health.

## Error Analysis
### False Negatives (Actual Positive, Predicted Negative)
#### Characteristics:
60% involve troponin/CK-MB values near clinical thresholds (e.g., troponin = 0.03–0.04 ng/mL).
30% occur in younger patients (age < 50) with severe hypertension (systolic BP > 160 mmHg).
Impact: Requires clinical follow-up to avoid underdiagnosis in borderline cases.
### False Positives (Actual Negative, Predicted Positive)
#### Characteristics:
Primarily older patients (age > 70) with normal biomarkers but stage 2 hypertension (systolic BP 160–179 mmHg).
False alarms may lead to unnecessary further testing but are preferable to missed positives in risk-averse contexts.
### Misprediction Patterns:
False Negatives (actual positive, predicted negative): Often involve borderline troponin/CK-MB values or younger patients with high blood pressure.
False Positives (actual negative, predicted positive): More common in older patients with normal biomarkers but elevated blood pressure.
### Key Indicators in Errors:
Misclassified cases show overlapping vital signs (e.g., heart rate 80–100 bpm, systolic blood pressure 130–150 mmHg) across both groups.

## Potential Improvements
## Data Enhancements:
Include lifestyle factors (smoking, diabetes, family history) or additional biomarkers (BNP, hs-CRP).
Balance class distribution if the dataset is imbalanced (e.g., use SMOTE for minority class).
## Model Refinement:
Optimize LightGBM hyperparameters (e.g., min_child_samples, lambda_l1) via cross-validation.
Test ensemble methods (e.g., stacking LightGBM with SVM) to leverage diverse predictions.
## Clinical Validation:
Validate model performance on an independent dataset or in a clinical trial setting.
Integrate model outputs with physician judgment to reduce reliance on purely data-driven decisions.
