# About 
This repository contains notebooks I've used to learn and explore machine learning concepts.

# Content
1. [Car Price Prediction and Obesity Classification](car_price_prediction_and_obesity_classification/car_price_prediction_and_obesity_classification.ipynb)
Collected a [car dataset from Kaggle](https://www.kaggle.com/datasets/erolmasimov/price-prediction-multiple-linear-regression), pre-processed the data using **pandas**, performed exploratory data analysis using **seaborn** and **matploltib**, encoded the data using **LabelEncoder**, split the dataset using **sklearn**, created a **linear regression**, **random forest**, and a **gradient boosting regressor** model, evaluated the models using **mean absolute error** and **r-squared**.
Collected an [obesity or CVD risk dataset from Kaggle](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster), pre-processed the data using **pandas**, performed exploratory data analysis using **seaborn** and **matploltib**, encoded the data using **LabelEncoder**, split the dataset using **sklearn**, created a decision tree, K-nearest neighbors, and **Random Forest classifier**, evaluated the models using **precision**, **recall**, and **F1 score**.

1. [Machine Learning Classification](classification/classification.ipynb)
Used the breast cancer dataset available through **sklearn.datasets**, split the dataset into train and test splits (80/20), performed exploratory data analysis with **seaborn**, trained a logistic regression model, and evaluated its performance with the calculated **precision**, **recall**, and **F1 score**.
Used the iris dataset available through **sklearn.datasets**, split the dataset into train and test splits (80/20), performed exploratory data analysis using **seaborn**, trained a **K-nearest neighbors classifier**, evaluated the performance of the model using **precision**, **recall**, and **F1 score**.

1. [Decision Theory in Machine Learning](decision_theory_in_ml/decision_theory_in_ml.ipynb)
Explored decision theory in a smartphone choice scenario based on criteria such as price, battery life, and camera quality.

1. [Information Theory in Machine Learning](information_theory_in_ml/information_theory_in_ml.ipynb)
Used the iris and the wine datasets available through **sklearn**, calculated **entropy**, calculated information gain.

1. [Probability in Machine Learning](probability_in_ml/probability_in_ml.ipynb)
Created a basic **Bayesian** email **classifier** to differentiate between spam and ham emails using an email dataset.

1. [Introduction to **PyTorch**](pytorch/pytorch.ipynb)
Explored tensors, **PyTorch** operations, **torch.autograd**, **gradients**, **backpropagation**, and built a simple **PyTorch** model.
Created a **PyTorch** model with two hidden layers, trained the model, evaluated the model, and plotted a decision boundary.

1. [Regression](regression/regression.ipynb)
Generated a synthetic data of hours studied vs exam score using **numpy**, visualized the dataset using **matplotlib**, split the data into train and test sets, created and trained a **Linear Regression model**, evaluated the model with **mean squared error** and **r-squared**, and plotted the prediction line against the ground truth.

Performed the same operations using an [existing salary prediction dataset from Kaggle](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression)

1. [Social Media and Mental Health Data Analysis and Visualization](social_media_mental_health/impact_of_social_media_on_mental_health.ipynb)
Collected a [mental health and social media impact dataset from Kaggle](https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health/data), cleaned and pre-processed the dataset using **pandas**, performed exploratory data analysis using **matplotlib** and **seaborn**, formulated hypotheses based on the findings, performed statistical analysis using chi-square and Pearson correlation tests, and visualized the findings.

1. [Introduction to **TensorFlow**](tensorflow/tensorflow.ipynb)
Created tensors, performed operations on tensors, and built a simple **TensorFlow** model.
Generated synthetic data, built a simple **TensorFlow** model, trained the model, and tested the model.

1. [AutoML with AutoGluon](automl/AutoML_AutoGluon.ipynb)
Explored AutoML by using AutoGluon for a classification problem (Salary prediction [<=50K/>50K])