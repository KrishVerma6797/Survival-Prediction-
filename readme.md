# Titanic Survival Prediction using Machine Learning

## 📌 Project Overview
This project predicts whether a passenger survived the Titanic disaster using basic Machine Learning techniques.  
It is an end-to-end beginner ML project covering data analysis, preprocessing, model training, evaluation, and prediction.

The goal of this project is to understand how a complete Machine Learning pipeline works in practice.

---

## 📊 Dataset
- Dataset: Titanic Dataset (CSV file)
- Rows: ~890 passengers
- Target Variable: `Survived`
  - `1` → Survived
  - `0` → Did not survive

### Features Used
| Feature | Description |
|------|------------|
| Age | Passenger age |
| Fare | Ticket fare |
| Sex | Gender (male/female) |
| Pclass | Passenger class (1, 2, 3) |

---

## 🧠 Machine Learning Workflow

### 1. Data Loading
- Loaded data using **Pandas**
- Selected only relevant columns

### 2. Data Understanding & EDA
- Checked missing values
- Visualized Age and Fare distributions
- Used histograms and QQ plots to understand data skewness

### 3. Data Preprocessing
- Missing values handled using **SimpleImputer**
- Log transformation applied to numerical features
- Categorical feature (`Sex`) encoded using **OneHotEncoder**
- Used **ColumnTransformer** to combine transformations

### 4. Models Used
- **Logistic Regression**
- **Decision Tree Classifier**

Both models were trained using a **Pipeline**, ensuring preprocessing and modeling happen together.

### 5. Model Evaluation
- Accuracy score on test data
- 10-fold cross-validation for better reliability

---

## 📈 Results (Approximate)
| Model | Accuracy |
|-----|---------|
| Logistic Regression | ~78% |
| Decision Tree | ~75% |
| Logistic Regression (CV) | Stable |
| Decision Tree (CV) | Slightly lower due to overfitting |

---

## 🧪 User Prediction
The model can predict survival based on user input:
- Age
- Fare
- Sex
- Passenger class

Example output:
- Passenger is likely to SURVIVE
- Passenger is NOT likely to survive

---

## 🗂️ Project Structure
