# =========================
# 1. Imports
# =========================
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# =========================
# 2. Data Collection & Loading
# =========================
df = pd.read_csv("titanic.csv")

df = df[['Survived', 'Age', 'Fare', 'Sex', 'Pclass']]

# =========================
# 3. Understanding the Data
# =========================
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df['Survived'].value_counts())

# =========================
# 4. Exploratory Data Analysis (EDA)
# =========================
plt.figure(figsize=(12,4))
plt.subplot(121)
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")

plt.subplot(122)
stats.probplot(df['Age'].dropna(), dist="norm", plot=plt)
plt.title("Age QQ Plot")
plt.show()

plt.figure(figsize=(12,4))
plt.subplot(121)
sns.histplot(df['Fare'], kde=True)
plt.title("Fare Distribution")

plt.subplot(122)
stats.probplot(df['Fare'], dist="norm", plot=plt)
plt.title("Fare QQ Plot")
plt.show()

# =========================
# 5. Feature Engineering
# =========================
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. Feature Transformation Pipeline
# =========================
numeric_features = ['Age', 'Fare']
categorical_features = ['Sex']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('log', FunctionTransformer(np.log1p))
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='passthrough')

# =========================
# 7. Model Pipelines
# =========================
lr_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

dt_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', DecisionTreeClassifier(random_state=42))
])

# =========================
# 8. Model Training
# =========================
lr_pipeline.fit(X_train, y_train)
dt_pipeline.fit(X_train, y_train)

# =========================
# 9. Evaluation
# =========================
y_pred_lr = lr_pipeline.predict(X_test)
y_pred_dt = dt_pipeline.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# =========================
# 10. Cross Validation (Correct Way)
# =========================
lr_cv = cross_val_score(lr_pipeline, X, y, cv=10, scoring='accuracy').mean()
dt_cv = cross_val_score(dt_pipeline, X, y, cv=10, scoring='accuracy').mean()

print("LR CV Accuracy:", lr_cv)
print("DT CV Accuracy:", dt_cv)

# ===============================
# USER INPUT & PREDICTION
# ===============================

# Take input from user
age = float(input("Enter Age: "))
fare = float(input("Enter Fare: "))
sex = input("Enter Sex (male/female): ").lower()
pclass = int(input("Enter Passenger Class (1/2/3): "))

# Create input DataFrame (same structure as training data)
user_input = pd.DataFrame({
    'Age': [age],
    'Fare': [fare],
    'Sex': [sex],
    'Pclass': [pclass]
})

# Predict using Logistic Regression pipeline
prediction = lr_pipeline.predict(user_input)[0]

# Output result
if prediction == 1:
    print("✅ Passenger is likely to SURVIVE")
else:
    print("❌ Passenger is NOT likely to survive")


