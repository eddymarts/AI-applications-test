# =============================================================================
#     Test of Regression algorithms for datasets:
#      1. Motor Parkinson Score
#      2. Total Parkinson Score
#
#     Algorithms used  
#      1. K Nearest Neighbours (with Optimum number of Neighbours)
#      2. Decision Trees Regressor
#      3. Random Forests Regressor
#      4. Support Vector Machines Regressor
#      5. Linear Regression
#     
#     Performance of each algorithm is computed by the square root of the
#     Mean Squared Error (MSE).
# 
#    Coded by Eddy Martínez
#             20209153
#    MSc Robotics Engineering, Liverpool Hope University
# =============================================================================

import pandas as pd
import numpy as np

#Import Parkinson dataset
Par_df=pd.read_csv('UPDRS.csv', sep=',',header=None)
Par_X_pd = Par_df.drop(axis=0, index=0, columns=[0, 20, 21])
Par_Y_pd = Par_df.drop(axis=0, index=0, columns=range(0,20))

Par_X = np.matrix(Par_X_pd)
Par_Y1 = np.array(Par_Y_pd).T[0,:]
Par_Y2 = np.array(Par_Y_pd).T[1,:]
Par_X = Par_X.astype(np.float)
Par_Y1 = Par_Y1.astype(np.float)
Par_Y2 = Par_Y2.astype(np.float)

n = 10
from RegressorComparison import CompareRegressor
Par_perf1 = CompareRegressor(Par_X, Par_Y1, n)
Par_perf2 = CompareRegressor(Par_X, Par_Y2, n)

Results = pd.DataFrame({'Motor UPDRS': Par_perf1,
                        'Total UPDRS': Par_perf2},
                        index=['KNN', 'DT', 'RF', 'SVM', 'LR'])
print(Results)