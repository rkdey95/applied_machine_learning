# Epilepsy Detection Using Applied_machine_learning

Created by: Rupesh Kumar Dey
Date 30-Apr-2021
Language: R-programming

This repository contains Scripts of Applied Machine Learning in R-programming language for Epilepsy Classification. 
Models built here include 
a) Support Vector Machines
b) Artificial Nueral Network 
c) KNN classifier

Files in the repository include:
1) data_raw.csv
- Raw dataset of EEG reading which is in the form of a numerical time series dataset of brain activity voltage readings over a period of time
- Y values indicate whether the individual has Epilepsy or not.

2) Dataet Preparation
- Script written to extract multiple non-linear statistical features from the raw dataset which is a non-linear time series
- The features extracted for each data entry will be used as dataset features for the Machine Learning Model. 
- Features extracted include Approximate Entropy, Sample Entropy, Hurst Exponent etc. 
- The features are saved in a structured format and writted / saved in the project directory in .csv format. 

3) SVM model
- Script that develops the SVM model used for classification.
- 3 Different SVM models were trained and tested and the best model with best hyperparemeters was selected

4) ANN model
- Script that develops the ANN model used for classification.
- 3 Different ANN models were trained and tested and the best model with best hyperparemeters was selected

5) KNNC Model
- Script that develops the KNN model used for classification.
- 3 Different KNN models were trained and tested and the best model with best hyperparemeters was selected

6) final model comparison
- Script written to test the performance of the best ANN, SVM and KNNC model to determine which is the best performing.

7) Final Report
- The final report detailing the study. Official Documentation. 
