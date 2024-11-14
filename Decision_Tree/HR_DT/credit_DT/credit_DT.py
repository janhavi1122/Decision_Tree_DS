# -- coding: utf-8 --
"""
Created on Fri Feb  2 08:35:57 2024

@author: janhavi
''
Business Problem
The company aims to assess credit risk based on customer information, such as balance, credit history, and employment details, to predict the likelihood of default. This information can then guide credit approval decisions and help the business reduce financial risk by categorizing customers as likely to default or not.

Business Objective
The main objective of this model is to build a decision tree classifier to predict customer default. By predicting credit risk, the company can:

Minimize potential loan defaults and financial losses.
Improve loan approval processes by automating risk assessment.
Enhance overall decision-making in granting credit, benefiting both the business and responsible borrowers.
Constraints
Data Quality Constraints: Limited or missing customer information can impact model accuracy.
Model Interpretability: The model should be easily interpretable so credit officers can understand the basis of credit decisions.
Regulatory Compliance: The model needs to comply with regulations regarding fair credit practices and non-discriminatory decision-making.
Scalability: The solution should be scalable to handle large volumes of data and adapt to new data points over time.
Bias Reduction: Efforts are needed to minimize potential biases in the model, ensuring fair treatment across different demographics. 
'''
"""
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import LabelEncoder
###########################################################

#C:/Decision Tree/credit.csv
data=pd.read_csv("E:/datascience/Decision_Tree/credit1.csv")
data.isnull().sum()
data.dropna()
data.info()
data=data.drop(['phone'],axis=1)
###########################################################

#Conconverting into Binary
lb=LabelEncoder()
data['checking_balance']=lb.fit_transform(data['checking_balance'])
data['credit_history']=lb.fit_transform(data['credit_history'])
data['purpose']=lb.fit_transform(data['purpose'])
data['other_credit']=lb.fit_transform(data['other_credit'])
data['housing']=lb.fit_transform(data['housing'])
data['savings_balance']=lb.fit_transform(data['savings_balance'])
data['employment_duration']=lb.fit_transform(data['employment_duration'])
data['job']=lb.fit_transform(data['job'])
###########################################################

#below 3 lines very imp used in every model
data['default'].unique()
data['default'].value_counts
colnames=list(data.columns)
predictors=colnames[:15]
target=colnames[15]
###########################################################

from sklearn.model_selection import train_test_split
#train the data
train,test= train_test_split(data,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT 

model=DT(criterion='entropy')
model.fit(train[predictors],train[target])
preds=model.predict(test[predictors])
preds

pd.crosstab(test[target],preds,rownames=['Actual'],colnames=['Predicted'])
np.mean(preds==test[target])

#########################################################################

'''
3. Data Pre-processing
Data Cleaning:

Remove any null values (as done in your code).
Drop irrelevant columns (like phone) to focus on predictive features.
Feature Engineering:

Label encode categorical features (e.g., checking_balance, purpose).
You can also standardize or normalize continuous variables if present (using StandardScaler or MinMaxScaler).
###########################################################

4. Exploratory Data Analysis (EDA)

Summary
Check summary statistics (data.describe()) for numerical features.
Explore the distribution of target classes (default) to understand class imbalance.
4.2 Univariate Analysis
Plot each feature’s distribution using sns.countplot for categorical variables.
Identify outliers using box plots for continuous variables.

Bivariate Analysis
Correlations: Use a heatmap to check correlation among features and identify relationships with the target.
Cross-tabulations: Compare categorical features against the target variable to understand their impact.
###########################################################

5. Model Building
Build the Model on Scaled Data
Split the data into training and testing sets.
Standardize if needed for models like Random Forest (not required for Decision Tree but can help in Random Forest).

Decision Tree and Random Forest Implementation
Decision Tree: Train the Decision Tree Classifier on training data using criterion='entropy'.

Random Forest: Train the Random Forest Classifier with similar parameters to Decision Tree for comparison.
###########################################################

Train, Test, and Cross-Validation
Split your data into training and testing sets.
Apply k-fold cross-validation to evaluate model stability across different subsets.
Track metrics like accuracy, precision, recall, and F1-score.
###########################################################

Documentation of Model Output
Decision Tree: Record feature importance and interpret the decision rules generated.
Random Forest: Evaluate feature importance and overall accuracy.
Document the models’ results, including confusion matrix and key metrics, to explain performance.
###########################################################

6. Benefits/Impact of the Solution
The models can help the company identify high-risk customers who are more likely to default, which allows for more informed credit-lending decisions. By accurately predicting potential defaulters, the company can:

Reduce financial risk: Minimize losses by carefully managing high-risk clients.
Improve customer profiling: Tailor loan products according to customer credit profiles.
Enhance operational efficiency: Focus resources on high-value, low-risk customers.
This solution provides a data-driven approach to improving lending decisions and ultimately strengthens the company’s risk management strategy.
'''
