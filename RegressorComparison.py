# =============================================================================
#     Comparison of Regression algorithms
#         -K Nearest Neighbours (with Optimum number of Neighbours)
#         -Gaussian Naïve Bayes Classifier
#         -Decision Trees
#         -Support Vector Machines
#         -Logistic Regression
#     
#     Performance of each algorithm is computed by Standard Deviation.
# 
#    Coded by Eddy Martínez
#             20209153
#    MSc Robotics Engineering, Liverpool Hope University
# =============================================================================

def CompareRegressor(X, Y, n):
    from DataProcessing import ScaleSplit
    from KNearestNeighbours import ONRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    from Performance import Pred_SD
    
    rg_perf = [0]*5
    
    for i in range(0,n):
        X_train, X_test, Y_train, Y_test = ScaleSplit(X, Y, 0.2)
        
        Y_knn = ONRegressor(X_train, X_test, Y_train, Y_test)
        
        dtree = DecisionTreeRegressor().fit(X_train, Y_train)
        Y_dt = dtree.predict(X_test)
        
        rf = RandomForestRegressor().fit(X_train, Y_train)
        Y_rf = rf.predict(X_test)
        
        SVM = SVR().fit(X_train, Y_train)
        Y_svm = SVM.predict(X_test)
        
        lr = LinearRegression().fit(X_train, Y_train)
        Y_lr = lr.predict(X_test)
        
        Predictions = {'Actual data': Y_test,
                        'KNN': Y_knn,
                        'DT': Y_dt,
                        'RF': Y_rf,
                        'SVM': Y_svm,
                        'LR': Y_lr}
        
        import pandas as pd
        Predictions_df = pd.DataFrame(Predictions)
        print(Predictions_df)
        
        rg_perf[0] += Pred_SD(Y_test, Y_knn)/n
        rg_perf[1] += Pred_SD(Y_test, Y_dt)/n
        rg_perf[2] += Pred_SD(Y_test, Y_rf)/n
        rg_perf[3] += Pred_SD(Y_test, Y_svm)/n
        rg_perf[4] += Pred_SD(Y_test, Y_lr)/n
        
    return rg_perf