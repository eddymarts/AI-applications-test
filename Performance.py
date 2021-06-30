# =============================================================================
#     Measure the performance of Classifiers and Regressors:
#     - Classifiers are measured by the percentage of predicted labels.
#     - Regressors are measured by Standard Deviation.
#                      
#     Coded by Eddy Martínez
#              20209153
#     MSc Robotics Engineering, Liverpool Hope University
# =============================================================================

def Pred_Percent(Y_test, Y_pred):
    count = 0;
    for i, j in zip(Y_test, Y_pred):
        if i==j:
            count+=1;
            
    return count*100/len(Y_test)


def Pred_SD(Y_test, Y_pred):
    import numpy as np
    SD = np.sqrt(np.sum((Y_test-Y_pred)**2)/len(Y_test))
            
    return SD
            