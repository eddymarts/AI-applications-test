# =============================================================================
#     Test of Classification algorithms for datasets:
#      1. Diabetes prediction
#
#     Algorithms used  
#      1. K Nearest Neighbours (with Optimum number of Neighbours)
#      2. Gaussian Naïve Bayes Classifier
#      3. Decision Trees
#      4. Random Forests
#      5. Support Vector Machines
#      6. Logistic Regression
#     
#     Performance of each algorithm is computed by percentage of good
#     predictions.
# 
#    Coded by Eddy Martínez
#             20209153
#    MSc Robotics Engineering, Liverpool Hope University
# =============================================================================

import pandas as pd
import numpy as np

#Import Diabetes dataset
Dia_df=pd.read_csv('Diabetes_num.csv', sep=',',header=None)
Dia_X_pd = Dia_df.drop(axis=0, index=0, columns=16)
Dia_Y_pd = Dia_df.drop(axis=0, index=0, columns=range(0,16))

Dia_X = np.matrix(Dia_X_pd)
Dia_Y = np.array(Dia_Y_pd).T[0,:]

#Import HAR dataset
HAR_df=pd.read_csv('HAR.csv', sep=',',header=None)
HAR_X_pd = HAR_df.drop(axis=0, index=0, columns=[0,4])
HAR_Y_pd = HAR_df.drop(axis=0, index=0, columns=range(0,4))

HAR_X = np.matrix(HAR_X_pd)
HAR_Y = np.array(HAR_Y_pd).T[0,:]

n = 1
from ClassifierComparison import CompareClassifier
Dia_perf = CompareClassifier(Dia_X, Dia_Y, n, 5)
HAR_perf = CompareClassifier(HAR_X, HAR_Y, n, 55)

Results = pd.DataFrame({'Diabetes': Dia_perf,
                        'HAR': HAR_perf},
                         index=['KNN', 'NBC', 'DT', 'RF', 'SVM', 'Log'])
print(Results)

