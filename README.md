# About 
This repository contains notebooks I've used to learn and explore machine learning concepts.

# Content
- [Car Price Prediction and Obesity Classification](car_price_prediction_and_obesity_classification/car_price_prediction_and_obesity_classification.ipynb)
1. Collected a [car dataset from Kaggle](https://www.kaggle.com/datasets/erolmasimov/price-prediction-multiple-linear-regression), pre-processed the data using **pandas**, performed exploratory data analysis using **seaborn** and **matploltib**, encoded the data using **LabelEncoder**, split the dataset using **sklearn**, created a **linear regression**, **random forest**, and a **gradient boosting regressor** model, evaluated the models using **mean absolute error** and **r-squared**.
1. Collected an [obesity or CVD risk dataset from Kaggle](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster), pre-processed the data using **pandas**, performed exploratory data analysis using **seaborn** and **matploltib**, encoded the data using **LabelEncoder**, split the dataset using **sklearn**, created a decision tree, K-nearest neighbors, and **Random Forest classifier**, evaluated the models using **precision**, **recall**, and **F1 score**.

- [Machine Learning Classification](classification/classification.ipynb)
1. Used the breast cancer dataset available through **sklearn.datasets**, split the dataset into train and test splits (80/20), performed exploratory data analysis with **seaborn**, trained a logistic regression model, and evaluated its performance with the calculated **precision**, **recall**, and **F1 score**.
1. Used the iris dataset available through **sklearn.datasets**, split the dataset into train and test splits (80/20), performed exploratory data analysis using **seaborn**, trained a **K-nearest neighbors classifier**, evaluated the performance of the model using **precision**, **recall**, and **F1 score**.

- [Decision Theory in Machine Learning](decision_theory_in_ml/decision_theory_in_ml.ipynb)
Explored decision theory in a smartphone choice scenario based on criteria such as price, battery life, and camera quality.

- [Information Theory in Machine Learning](information_theory_in_ml/information_theory_in_ml.ipynb)
Used the iris and the wine datasets available through **sklearn**, calculated **entropy**, calculated information gain.

- [Probability in Machine Learning](probability_in_ml/probability_in_ml.ipynb)
Created a basic **Bayesian** email **classifier** to differentiate between spam and ham emails using an email dataset.

- [Introduction to **PyTorch**](pytorch/pytorch.ipynb)
1. Explored tensors, **PyTorch** operations, **torch.autograd**, **gradients**, **backpropagation**, and built a simple **PyTorch** model.
1. Created a **PyTorch** model with two hidden layers, trained the model, evaluated the model, and plotted a decision boundary.

- [Regression](regression/regression.ipynb)
1. Generated a synthetic data of hours studied vs exam score using **numpy**, visualized the dataset using **matplotlib**, split the data into train and test sets, created and trained a **Linear Regression model**, evaluated the model with **mean squared error** and **r-squared**, and plotted the prediction line against the ground truth.
1. Performed the same operations using an [existing salary prediction dataset from Kaggle](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression)

- [Social Media and Mental Health Data Analysis and Visualization](social_media_mental_health/impact_of_social_media_on_mental_health.ipynb)
Collected a [mental health and social media impact dataset from Kaggle](https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health/data), cleaned and pre-processed the dataset using **pandas**, performed exploratory data analysis using **matplotlib** and **seaborn**, formulated hypotheses based on the findings, performed statistical analysis using chi-square and Pearson correlation tests, and visualized the findings.

- [Introduction to **TensorFlow**](tensorflow/tensorflow.ipynb)
1. Created tensors, performed operations on tensors, and built a simple **TensorFlow** model.
1. Generated synthetic data, built a simple **TensorFlow** model, trained the model, and tested the model.