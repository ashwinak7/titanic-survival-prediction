# Titanic Survival Prediction Using Logistic Regression

This project demonstrates the use of Logistic Regression to predict passenger survival on the Titanic dataset. The model is trained on historical passenger data with features such as age, sex, and passenger class, and evaluated with various performance metrics.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Test Set Prediction](#test-set-prediction)
- [Results](#results)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [File Structure](#file-structure)
- [Author](#author)

---

## Project Overview

The goal of this project is to build a Logistic Regression model to classify whether a passenger survived or not on the Titanic. The project includes:

- Data cleaning and feature engineering.
- Visualization of survival counts and survival by gender.
- Splitting the data into training and validation sets.
- Training and evaluating a Logistic Regression model.
- Visualizing performance using confusion matrix and ROC curve.
- Predicting survival on the test dataset.

---

## Dataset

The project uses the popular Titanic dataset from Kaggle:

- `train.csv`: Contains passenger data with the target variable `Survived`.
- `test.csv`: Contains passenger data without survival information for prediction.

Features used include:
- `Pclass` (Passenger class)
- `Sex` (converted to numeric)
- `Age`
- `SibSp` (Number of siblings/spouses aboard)
- `Parch` (Number of parents/children aboard)
- `Fare`

Dropped features:
- `Name`, `Ticket`, `Cabin`, `Embarked`

---

## Data Preprocessing

- Dropped irrelevant columns (`Name`, `Embarked`, `Ticket`, `Cabin`).
- Mapped `Sex` from categorical (`male`/`female`) to numeric (0/1).
- Handled missing values by dropping rows in training and imputing with mean in test data.

---

## Exploratory Data Analysis (EDA)

Two key visualizations were created:

1. **Survival Count:** Shows the number of survivors vs. non-survivors.
2. **Survival by Gender:** Compares survival counts between males and females.

Bar charts include annotated counts for easy interpretation.

---

## Model Training and Evaluation

- Split training data into 80% training and 20% validation subsets with stratification.
- Trained Logistic Regression with max iterations set to 200.
- Evaluated using:
  - Accuracy
  - Classification Report (Precision, Recall, F1-score)
  - Confusion Matrix (with heatmap visualization)
  - ROC AUC score and ROC curve plot

---

## Test Set Prediction

- Preprocessed test data with the same steps as training data.
- Made survival predictions using the trained model.
- Outputted the first 10 predictions as an example.

---

## Results

- **Validation Accuracy:** Approximately 80.4%
- **ROC AUC Score:** Approximately 0.86
- Confusion matrix and classification report indicate balanced performance across classes.

---

## How to Run

1. Clone or download the repository.
2. Ensure `train.csv` and `test.csv` are placed in the specified dataset folder.
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
