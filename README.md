# Breakaway_win
Project Overview
This project aims to develop a machine learning model to predict whether an early breakaway will take the win in a cycling Grand Tour stage. The train data used for this project is taken from the years 2013 to 2022 and the test data from 2023 (waiting for Vuelta result). The project includes data preparation, model selection, cross-validation, and evaluation of different machine learning algorithms.

# Librairies used
pandas
numpy
scikit-learn
matplotlib

## 1. Overview of the data
First let's have a look at the number of breakaway win each year per Grand Tour :
![image](https://github.com/VioleauPierre/Breakaway_win/assets/129098391/9e81b505-cdda-430a-8bbe-f27b19ff2801)
There is between 4 and 10 breakaway win each year in Grand Tour, as the rider make the race, we can't predict the number of breakaway win in 2023 only based on that.

Now let's see if the stage classification have an impact on the number of breakaway win per year : 
![image](https://github.com/VioleauPierre/Breakaway_win/assets/129098391/780396fa-eea2-4a71-af43-f6f0fc032b0a)

Better than a number of stage win, a propoprtion of win for each stage classification will give us more informations :
![image](https://github.com/VioleauPierre/Breakaway_win/assets/129098391/cc5b46e4-fd7a-4ab0-b38d-82d341d08fed)
More than half of the hilly and mountain stages ended with a breakaway win.

## 2. Model Selection
Several machine learning algorithms are used to predict the "Breakaway win" based on the features. The following models are used:

Dummy classifier: A simple baseline that predicts the most frequent class.
Logistic regression
SVM (Support Vector Machine)
Decision trees
Random forest
Naive Bayes
K-Nearest Neighbors (KNN)
Model Evaluation
Each model is evaluated using cross-validation to estimate its performance. The evaluation metrics used are accuracy score and the Receiver Operating Characteristic (ROC) curve. The area under the ROC curve (AUC) is used to measure the model's discriminative ability.

## 3. Optimizing the Best Model (KNN)
Based on the initial evaluation, KNN showed the best performance. Hyperparameter tuning is performed using Exhaustive Grid Search to find the best value for the number of neighbors (k). The hyperparameters metric, n_neighbors, and weights are optimized. After tuning, the model is evaluated again and the results are compared.

## 4. Results
The results for all models are summarized in a DataFrame df_result. The predictions for the "Breakaway_win" made by KNN before and after optimization are included for each stage of the Giro and the Tour. !It only contain Giro and Tour (waiting for Vuelta result)!
