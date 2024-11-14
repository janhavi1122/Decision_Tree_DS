# -- coding: utf-8 --
"""
Created on Fri Feb  2 08:35:57 2024

@author: janhavi

Business Problem
The company aims to assess credit risk based on customer information, such as balance, credit history, and employment details, to predict the likelihood of default. This information can then guide credit approval decisions and help the business reduce financial risk by categorizing customers as likely to default or not.

Business Objective
The main objective of this model is to build a decision tree classifier to predict customer default. By predicting credit risk, the company can:

1. Minimize potential loan defaults and financial losses.
2. Improve loan approval processes by automating risk assessment.
3. Enhance overall decision-making in granting credit, benefiting both the business and responsible borrowers.

Constraints
- Data Quality Constraints: Limited or missing customer information can impact model accuracy.
- Model Interpretability: The model should be easily interpretable so credit officers can understand the basis of credit decisions.
- Regulatory Compliance: The model needs to comply with regulations regarding fair credit practices and non-discriminatory decision-making.
- Scalability: The solution should be scalable to handle large volumes of data and adapt to new data points over time.
- Bias Reduction: Efforts are needed to minimize potential biases in the model, ensuring fair treatment across different demographics.
"""

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load Data
data = pd.read_csv("E:/datascience/Decision_Tree/credit1.csv")
data = data.drop(['phone'], axis=1)  # Drop irrelevant columns

# Check for and handle missing values
data.dropna(inplace=True)

# Encoding categorical features to binary values
lb = LabelEncoder()
for col in ['checking_balance', 'credit_history', 'purpose', 'other_credit', 'housing', 'savings_balance', 'employment_duration', 'job']:
    data[col] = lb.fit_transform(data[col])

# Define predictors and target
colnames = list(data.columns)
predictors = colnames[:15]
target = colnames[15]

# Split data into training and testing sets
train, test = train_test_split(data, test_size=0.3)

# Model Building: Decision Tree Classifier
model = DecisionTreeClassifier(criterion='entropy')
model.fit(train[predictors], train[target])
preds = model.predict(test[predictors])

# Model Evaluation
confusion_matrix = pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predicted'])
accuracy = np.mean(preds == test[target])

# Display Confusion Matrix and Accuracy
print("Confusion Matrix:\n", confusion_matrix)
print("\nAccuracy:", accuracy)

#########################################################################

'''
3. Data Pre-processing
- Data Cleaning: Remove null values and irrelevant columns.
- Feature Engineering: Label encode categorical features and standardize/normalize if required.

4. Exploratory Data Analysis (EDA)
Summary
- Check summary statistics (data.describe()) for numerical features.
- Explore target class distribution to understand class imbalance.

Univariate Analysis
- Plot distributions for categorical variables and identify outliers using box plots.

Bivariate Analysis
- Check correlations with a heatmap and use cross-tabulations to understand feature relationships with the target.

5. Model Building
Build the Model on Scaled Data
- Train/test split and scale data if required.

Model Selection
- Decision Tree (using entropy criterion)
- Compare with Random Forest if needed.

Train, Test, and Cross-Validation
- k-fold cross-validation to evaluate model stability and track metrics (accuracy, precision, recall, F1-score).

Documentation of Model Output
- Record feature importance, interpret decision rules, and evaluate confusion matrix.

6. Benefits/Impact of the Solution
This solution helps the company identify high-risk customers, reducing financial risk, improving customer profiling, and enhancing operational efficiency. This data-driven approach strengthens the company’s risk management strategy.
'''
