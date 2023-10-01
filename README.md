# Breakaway_win
## Project Overview : 
This project aims to develop a machine learning model to predict whether an early breakaway will take the win in a cycling Grand Tour stage. The train data used for this project is taken from the years 2013 to 2022 and the test data from 2023 (waiting for Vuelta result). The project includes data preparation, model selection, cross-validation, and evaluation of different machine learning algorithms.

## Librairies used
- pandas
- numpy
- scikit-learn
- matplotlib

## Skills : 
- Machine Learning Models: Logistic Regression, SVM, Decision Trees, Random Forest, Naive Bayes, and K-Nearest Neighbors (KNN).
- Model Selection and Evaluation: The project uses cross-validation to select the best-performing model and evaluates the models using accuracy scores and Receiver Operating Characteristic (ROC) curves to measure their predictive performance.
- Hyperparameter Tuning: Hyperparameter tuning is demonstrated using GridSearchCV to optimize the KNN model's performance by finding the best values for hyperparameters.

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

- Dummy classifier: A simple baseline that predicts the most frequent class.
- Logistic regression
- SVM (Support Vector Machine)
- Decision trees
- Random forest
- Naive Bayes
- K-Nearest Neighbors (KNN)
Model Evaluation
Each model is evaluated using cross-validation to estimate its performance. The evaluation metrics used are accuracy score and the Receiver Operating Characteristic (ROC) curve. The area under the ROC curve (AUC) is used to measure the model's discriminative ability.

## 3. Optimizing the Best Model (KNN)
Based on the initial evaluation, KNN showed the best performance. Hyperparameter tuning is performed using Exhaustive Grid Search to find the best value for the number of neighbors (k). The hyperparameters metric, n_neighbors, and weights are optimized. After tuning, the model is evaluated again and the results are compared. Find below the confusion matrix from the tuned KNN model : ![image](https://github.com/VioleauPierre/Breakaway_win/assets/129098391/b2c54e7a-4e53-496c-b5c4-13440eaa89ba)


## 4. Results
The results for all models are summarized in a DataFrame df_result and in viewed usinig a bar plot. 
![image](https://github.com/VioleauPierre/Breakaway_win/assets/129098391/5375f503-fb40-4aaf-9777-5dd4baca4f06)

