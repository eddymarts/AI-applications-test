# =============================================================================
#     Comparison of Classification algorithms
#         -K Nearest Neighbours (with Optimum number of Neighbours)
#         -Gaussian Naïve Bayes Classifier
#         -Decision Trees
#         -Support Vector Machines
#         -Logistic Regression
#     
#     Performance of each algorithm is computed by percentage of good
#     predictions.
# 
#    Coded by Eddy Martínez
#             20209153
#    MSc Robotics Engineering, Liverpool Hope University
# =============================================================================

def CompareClassifier(X, Y, n, neighbors):
    from DataProcessing import ScaleSplit
    from KNearestNeighbours import ONClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    from Performance import Pred_Percent
    
    cl_perf = [0]*6
    
    for i in range(0,n):
        X_train, X_test, Y_train, Y_test = ScaleSplit(X, Y, 0.2)
        
        Y_knn = ONClassifier(X_train, X_test, Y_train, Y_test, neighbors)
        
        nbc = GaussianNB().fit(X_train, Y_train)
        Y_nbc = nbc.predict(X_test)
        
        dtree = tree.DecisionTreeClassifier().fit(X_train, Y_train)
        Y_dt = dtree.predict(X_test)
        
        rf = RandomForestClassifier().fit(X_train, Y_train)
        Y_rf = rf.predict(X_test)
        
        SVM = svm.SVC().fit(X_train, Y_train)
        Y_svm = SVM.predict(X_test)
        
        log = LogisticRegression(random_state=0).fit(X_train, Y_train)
        Y_log = log.predict(X_test)
        
        Predictions = {'Actual data': Y_test,
                        'KNN': Y_knn,
                        'NBC': Y_nbc,
                        'DT': Y_dt,
                        'RF': Y_rf,
                        'SVM': Y_svm,
                        'Log': Y_log}
        
        import pandas as pd
        Predictions_df = pd.DataFrame(Predictions)
        print(Predictions_df)
        
        cl_perf[0] += Pred_Percent(Y_test, Y_knn)/n
        cl_perf[1] += Pred_Percent(Y_test, Y_nbc)/n
        cl_perf[2] += Pred_Percent(Y_test, Y_dt)/n
        cl_perf[3] += Pred_Percent(Y_test, Y_rf)/n
        cl_perf[4] += Pred_Percent(Y_test, Y_svm)/n
        cl_perf[5] += Pred_Percent(Y_test, Y_log)/n
        
    return cl_perf