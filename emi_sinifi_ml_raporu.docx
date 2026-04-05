import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# 1) Veri setini oku
dataset = pd.read_csv("synthetic_personal_finance_dataset.csv")


# 2) Hedef değişkeni sınıfa çevir
# monthly_emi_usd:
#   0 ise no_loan
#   >0 olanlar kendi içinde low_emi / high_emi

positive_emi = dataset.loc[dataset["monthly_emi_usd"] > 0, "monthly_emi_usd"]
median_emi = positive_emi.median()

def emi_to_class(x):
    if x == 0:
        return "no_loan"
    elif x <= median_emi:
        return "low_emi"
    else:
        return "high_emi"

dataset["emi_class"] = dataset["monthly_emi_usd"].apply(emi_to_class)


# 3) Tarih sütununu parçala

dataset["record_date"] = pd.to_datetime(dataset["record_date"], errors="coerce")
dataset["record_year"] = dataset["record_date"].dt.year
dataset["record_month"] = dataset["record_date"].dt.month
dataset["record_dayofweek"] = dataset["record_date"].dt.dayofweek


# 4) Kullanılacak özellikler
# Not:
# user_id kullanılmadı
# monthly_emi_usd direkt hedefin kaynağı olduğu için X içine alınmadı

numeric_features = [
    "age",
    "monthly_income_usd",
    "monthly_expenses_usd",
    "savings_usd",
    "loan_amount_usd",
    "loan_term_months",
    "loan_interest_rate_pct",
    "debt_to_income_ratio",
    "credit_score",
    "savings_to_income_ratio",
    "record_year",
    "record_month",
    "record_dayofweek"
]

categorical_features = [
    "gender",
    "education_level",
    "employment_status",
    "job_title",
    "has_loan",
    "loan_type",
    "region"
]

X = dataset[numeric_features + categorical_features]
y = dataset["emi_class"]


# 5) veri ön işleme 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)



numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# 6) Model

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])




clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)


# 7) Performans çıktıları

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# 8) Confusion Matrix

cm = confusion_matrix(y_test, y_pred, labels=["no_loan", "low_emi", "high_emi"])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["no_loan", "low_emi", "high_emi"]
)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Confusion Matrix - EMI Class Prediction")
plt.show()