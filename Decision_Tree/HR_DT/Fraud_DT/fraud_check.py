# -- coding: utf-8 --
"""
Created on Fri Feb  2 08:35:57 2024

@author: janhavi

Business Problem
The company aims to assess credit risk based on customer information, such as income, education, and employment details, to predict the likelihood of default. This information can guide credit approval decisions and help the business reduce financial risk by categorizing customers as likely to default or not.

Business Objective
The main objective of this model is to build a decision tree classifier to predict customer default based on features such as education and income. By predicting credit risk, the company can:

- Minimize potential loan defaults and financial losses.
- Improve loan approval processes by automating risk assessment.
- Enhance overall decision-making in granting credit, benefiting both the business and responsible borrowers.

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
################################################################

# Load the dataset
data = pd.read_csv("E:/datascience/Decision_Tree/credit_data.csv")  # Update the filename accordingly
################################################################

# Check for missing values
print(data.isnull().sum())
data.dropna(inplace=True)
################################################################

# Display data info
data.info()
################################################################

# Drop irrelevant columns if any (update this if needed)
# data = data.drop(['irrelevant_column'], axis=1)

# Converting categorical features into binary
label_encoders = {}
categorical_columns = ['Undergrad', 'Marital.Status', 'Urban']

for col in categorical_columns:
    lb = LabelEncoder()
    data[col] = lb.fit_transform(data[col])
    label_encoders[col] = lb

# Check unique values for target variable if applicable
print(data['Taxable.Income'].unique())
print(data['Taxable.Income'].value_counts())
################################################################

# Define predictors and target
colnames = list(data.columns)
predictors = colnames[:-1]  # All columns except the last one as predictors
target = colnames[-1]        # Last column as target
################################################################

# Train-test split
train, test = train_test_split(data, test_size=0.3, random_state=42)
################################################################

# Build the Decision Tree model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(train[predictors], train[target])
################################################################

# Make predictions
preds = model.predict(test[predictors])
################################################################

# Evaluate the model
confusion_matrix = pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

accuracy = np.mean(preds == test[target])
print(f'Accuracy: {accuracy:.2f}')
################################################################

# Exploratory Data Analysis (EDA)

# Summary statistics
print(data.describe())

# Visualizations
plt.figure(figsize=(10, 6))
sns.histplot(data['Taxable.Income'], kde=True)
plt.title('Histogram of Taxable Income')
plt.xlabel('Taxable Income')
plt.ylabel('Frequency')
plt.show()
################################################################

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Taxable.Income', data=data)
plt.title('Boxplot of Taxable Income')
plt.xlabel('Taxable Income')
plt.ylabel('Values')
plt.show()
################################################################

# Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='City.Population', y='Taxable.Income', data=data)
plt.title('Scatterplot of City Population vs. Taxable Income')
plt.xlabel('City Population')
plt.ylabel('Taxable Income')
plt.show()
################################################################

# Feature Importance
feature_importances = model.feature_importances_
features = pd.Series(feature_importances, index=predictors)
features.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.show()

################################################################
