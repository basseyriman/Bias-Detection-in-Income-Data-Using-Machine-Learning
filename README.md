# Introduction
This project focuses on utilizing machine learning to predict income bias in the Adult Income Dataset available on Kaggle. 
The goal is to build a model that predicts whether an individual's income is greater than or equal to $50K or less than $50K annually based on 
demographic information such as gender, age, work class, occupation, education, and race. By identifying potential biases in the income distribution, 
this model offers valuable insights into income inequality and the factors that influence it.

The dataset, sourced from Kaggle here, contains 15 columns, and is ideal for analyzing demographic factors contributing to income disparities. 
This project also explores the importance of bias detection in machine learning models, specifically focusing on factors such as race and gender.

# Key Features of the Project
Dataset: The dataset contains demographic attributes like age, work class, education, gender, and income class.

# Goal: 
Predict whether an individual’s income is greater than $50K using demographic features.

# Tools: 
The model was developed using Python, with libraries such as scikit-learn, pandas, matplotlib, and seaborn for data analysis and model building.

# Project Overview
Packages Import
import shap 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
sns.set_theme(style = 'whitegrid')

These libraries provide the necessary functionality for data processing, visualization, and machine learning model development.

# Loading the Data
The dataset is loaded using Pandas, and the first five rows are inspected:
adult = pd.read_csv("../Dataset/adult.csv")
adult.head()

# Sample data output:

age	workclass	fnlwgt	education	...	income
25	Private	226802	11th	...	<=50K
38	Private	89814	HS-grad	...	<=50K
28	Local-gov	336951	Assoc-acdm	...	>50K
44	Private	160323	Some-college	...	>50K

# Data Exploration and Visualization
Data exploration includes checking for missing values and generating visualizations such as:
plt.figure(figsize = (10, 6))
sns.countplot(x = 'gender', hue = 'education', data = adult, palette = 'YlGnBu')
plt.show()
Observation: Males tend to have more educational qualifications than females in the dataset.

# Data Cleaning and Preprocessing
Columns such as 'gender', 'education', 'income', and 'workclass' are replaced with numerical values for easier modeling:
adult["gender"].replace({'Male': 1, 'Female': 0}, inplace=True)
adult["education"].replace({ ... }, inplace=True)
adult["income"].replace({'<=50K': 0, '>50K': 1}, inplace=True)

Additional encoding is done for categorical variables like occupation, converting them into dummy variables.

# Model Training and Evaluation
After preprocessing the data, the dataset is split into features (X) and target (y):
X = adult.drop(columns=['marital-status', 'relationship', 'education', 'native-country'], axis=1)
y = adult["income"]

The model is trained using various algorithms such as Support Vector Machine (SVM), Naive Bayes, and evaluated using accuracy scores and SHAP values to analyze model predictions.
# Sample code for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC()
model.fit(X_train, y_train)

# Predicting and calculating accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Results and Observations
After training the model, the final weight and gender distribution for both income categories are visualized:
plt.figure(figsize = (10, 6))
sns.barplot(data = adult, y = 'fnlwgt', x = 'income', hue = 'gender', palette = 'YlGnBu')
plt.show()
Observation: The dataset reveals that men generally have higher weights and appear more often in the higher income group compared to women.

# Concluding Remarks
This project provides a machine learning-based approach to predict income bias and identify disparities in the distribution of income across various demographic groups. 
It also highlights the importance of addressing biases in datasets to improve fairness and equality in predictive modeling.

# Future Work
Addressing Bias: Future work could explore ways to mitigate bias in machine learning models, such as balancing the dataset or applying fairness constraints during model training.

Extended Analysis: Additional analysis on the effects of other demographic features such as marital status and occupation could be explored for further insights.

# Requirements
To run this project locally, ensure you have the following libraries installed:
pip install shap numpy pandas scikit-learn matplotlib seaborn tqdm

# Dataset
The dataset used in this project is available on Kaggle: Adult Income Dataset

# License
This project is licensed under the MIT License - see the LICENSE file for details.
