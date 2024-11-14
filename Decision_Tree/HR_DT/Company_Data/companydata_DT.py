# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:11:35 2024

@author: santo
"""
'''
Business Problem:
#What is the business objective?
Each problem has a unique business objective:
Problem 1: Help a cloth manufacturing company identify key attributes contributing to high sales.
Problem 2: Use diabetes data to predict patient outcomes based on various health factors.
Problem 3: Determine the risk level of individuals based on taxable income (e.g., Risky vs. Good).
Problem 4: Help HR verify salary claims based on factors like experience, role, etc 

#Are there any constraints?
Constraints in a business context can refer to various limitations, challenges, or restrictions that may impact the achievement of business objectives. The specific constraints depend on the nature of the business, industry, and external factors. 
'''

################################################################
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
###########################################################

# Load the Dataset
df = pd.read_csv("E:/datascience/Decision_Tree/Company_Data.csv.xls")
###########################################################

# Step 2: Feature Analysis (Data Dictionary)
# Displaying the data types and initial inspection
print(df.info())
print(df.describe())
print(df.head())
###########################################################

# Step 3: Data Pre-processing

# Data Cleaning - Checking for missing values
print("Missing values:\n", df.isnull().sum())
###########################################################

# Converting 'Sales' into a categorical variable (high/medium/low sales based on quantiles)
df['Sales'] = pd.qcut(df['Sales'], q=3, labels=['Low', 'Medium', 'High'])
###########################################################

# Encoding Categorical Variables
label_encoder = LabelEncoder()
df['ShelveLoc'] = label_encoder.fit_transform(df['ShelveLoc'])
df['Urban'] = label_encoder.fit_transform(df['Urban'])
df['US'] = label_encoder.fit_transform(df['US'])
###########################################################

# Feature Scaling - Scaling numeric features
scaler = StandardScaler()
numeric_features = ['CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age', 'Education']
df[numeric_features] = scaler.fit_transform(df[numeric_features])
###########################################################

# Step 4: Exploratory Data Analysis (EDA)

# Univariate Analysis - Histograms for numerical features
df[numeric_features].hist(bins=15, figsize=(15, 10))
plt.suptitle("Histograms of Numeric Features")
plt.show()
###########################################################

# Bivariate Analysis - Heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()
###########################################################

# Step 5: Model Building

# Splitting data into train and test sets
X = df.drop(columns=['Sales'])
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
###########################################################

# Model 1: Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
###########################################################

# Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
###########################################################

# Model Evaluation

# Decision Tree Evaluation
print("Decision Tree Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
###########################################################

# Random Forest Evaluation
print("\nRandom Forest Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
###########################################################

# Cross-Validation for both models
dt_scores = cross_val_score(dt_model, X, y, cv=5)
rf_scores = cross_val_score(rf_model, X, y, cv=5)

print("\nDecision Tree Cross-Validation Accuracy:", np.mean(dt_scores))
print("Random Forest Cross-Validation Accuracy:", np.mean(rf_scores))
###########################################################

# Step 6: Summary of Benefits
print("""
Benefits:
The insights gained from the model allow the business to understand which features impact sales the most.
This can guide pricing, marketing, and product placement strategies, ultimately boosting revenue and sales effectiveness.
""")
###########################################################

'''Data Preprocessing:

Missing values are checked, categorical variables are encoded, and numerical features are scaled.
Sales is converted into a categorical variable for classification purposes.
EDA:

Histograms are used for univariate analysis, and a heatmap of the correlation matrix is used for bivariate analysis.
Model Building:

Decision Tree and Random Forest models are created. Both models are trained and tested on the dataset.
Cross-Validation is used to get a robust accuracy estimate for each model.
Model Evaluation:

Key performance metrics include accuracy, classification report, and confusion matrix for both models.
Benefits:

The results highlight feature importance for sales prediction, which can guide data-driven decisions.
'''




