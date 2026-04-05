# EMI Class Prediction with Machine Learning

This project analyzes a synthetic personal finance dataset and builds a machine learning model to predict EMI classes.

## Project Goal
The original target variable `credit_score` did not show strong relationships with the available features.  
Therefore, the target was changed to `monthly_emi_usd`, and the problem was reformulated as a classification task.

## New Target
`monthly_emi_usd` was converted into 3 classes:
- `no_loan`
- `low_emi`
- `high_emi`

## Features Used
- age
- monthly_income_usd
- monthly_expenses_usd
- savings_usd
- loan_amount_usd
- loan_term_months
- loan_interest_rate_pct
- debt_to_income_ratio
- credit_score
- savings_to_income_ratio
- gender
- education_level
- employment_status
- job_title
- has_loan
- loan_type
- region
- record_year
- record_month
- record_dayofweek

## Methods
- Data preprocessing with `Pipeline`
- Missing value imputation
- Standardization for numeric variables
- One-hot encoding for categorical variables
- Random Forest Classifier

## Evaluation
The model was evaluated using:
- Accuracy
- Classification Report
- Confusion Matrix

## Output
The project includes:
- Word report
- Confusion matrix visualization
- Feature importance analysis

## Technologies
- Python
- pandas
- scikit-learn
- matplotlib

## Report
The detailed project report is available in the `reports/` folder.
