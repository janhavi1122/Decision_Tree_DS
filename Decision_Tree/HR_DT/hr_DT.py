# -- coding: utf-8 --
"""
Created on Fri Feb  2 08:35:57 2024

@author: janhavi
'''
Business Problem
The company aims to assess credit risk based on customer information, such as balance, credit history, and employment details, to predict the likelihood of default. This information can then guide credit approval decisions and help the business reduce financial risk by categorizing customers as likely to default or not.

Business Objective
The main objective of this model is to build a decision tree classifier to predict customer default. By predicting credit risk, the company can:
- Minimize potential loan defaults and financial losses.
- Improve loan approval processes by automating risk assessment.
- Enhance overall decision-making in granting credit, benefiting both the business and responsible borrowers.

Constraints
- Data Quality Constraints: Limited or missing customer information can impact model accuracy.
- Model Interpretability: The model should be easily interpretable so credit officers can understand the basis of credit decisions.
- Regulatory Compliance: The model needs to comply with regulations regarding fair credit practices and non-discriminatory decision-making.
- Scalability: The solution should be scalable to handle large volumes of data and adapt to new data points over time.
- Bias Reduction: Efforts are needed to minimize potential biases in the model, ensuring fair treatment across different demographics.
'''
"""

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("E:/datascience/Decision_Tree/credit1.csv")

# 2. Feature Information Table
# Assuming that the features are already defined in `data`
feature_info = {
    "Feature": data.columns,
    "Data Type": [str(data[feature].dtype) for feature in data.columns],
    "Relevance": [
        "Relevant for prediction" if feature != "default" else "Target variable" 
        for feature in data.columns
    ],
    "Description": [
        "Balance in the checking account" if feature == "checking_balance" else
        "Credit history status" if feature == "credit_history" else
        "Purpose of the loan" if feature == "purpose" else
        "Other credits status" if feature == "other_credit" else
        "Housing status" if feature == "housing" else
        "Savings account balance" if feature == "savings_balance" else
        "Employment duration in years" if feature == "employment_duration" else
        "Job type" if feature == "job" else
        "Binary target indicating default" if feature == "default" else
        "Other feature description"  # Placeholder for other features
        for feature in data.columns
    ]
}

feature_table = pd.DataFrame(feature_info)
print(feature_table)

# 3. Data Pre-processing
## 3.1 Data Cleaning
data.dropna(inplace=True)  # Remove rows with missing values
data.drop(columns=['phone'], axis=1, inplace=True)  # Drop irrelevant columns

# Feature Engineering
lb = LabelEncoder()
categorical_features = ['checking_balance', 'credit_history', 'purpose', 
                        'other_credit', 'housing', 'savings_balance', 
                        'employment_duration', 'job']
for feature in categorical_features:
    data[feature] = lb.fit_transform(data[feature])

# 4. Exploratory Data Analysis (EDA)
## 4.1 Summary
print(data.describe())  # Summary statistics

## 4.2 Univariate Analysis
plt.figure(figsize=(10, 6))
sns.countplot(x='default', data=data)
plt.title('Distribution of Default')
plt.xlabel('Default Status')
plt.ylabel('Frequency')
plt.show()

## 4.3 Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 5. Model Building
##Build the model on the scaled data
train, test = train_test_split(data, test_size=0.3, random_state=42)

##Perform Decision Tree and Random Forest
model_dt = DecisionTreeClassifier(criterion='entropy')
model_dt.fit(train.drop('default', axis=1), train['default'])
preds_dt = model_dt.predict(test.drop('default', axis=1))

# Cross-tabulation
print(pd.crosstab(test['default'], preds_dt, rownames=['Actual'], colnames=['Predicted']))
print("Accuracy:", np.mean(preds_dt == test['default']))

##Cross-validation and performance metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy = accuracy_score(test['default'], preds_dt)
precision = precision_score(test['default'], preds_dt)
recall = recall_score(test['default'], preds_dt)

print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

#Documentation of Model Output
# Record feature importance (if necessary) and interpret decision rules
print("Feature importances:", model_dt.feature_importances_)
#___________________________________________________________________________________

# 6. Benefits/Impact of the Solution
'''
The models can help the company identify high-risk customers who are more likely to default, which allows for more informed credit-lending decisions. By accurately predicting potential defaulters, the company can:
- Reduce financial risk: Minimize losses by carefully managing high-risk clients.
- Improve customer profiling: Tailor loan products according to customer credit profiles.
- Enhance operational efficiency: Focus resources on high-value, low-risk customers.
This solution provides a data-driven approach to improving lending decisions and ultimately strengthens the company’s risk management strategy.
'''
