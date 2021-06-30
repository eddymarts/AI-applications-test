# =============================================================================
#     K Nearest Neighbours Functions
#         -Classifier: Predicts the class of test data.
#         
#         -Regressor: Predicts the continuous output of test data.
#         
#         -ONClassifier: Searches for optimum number of neighbours to classify
#                        the data by plotting the error vs number of
#                        neighbours. The user select the number of neighbours
#                        after seeing the plot.
#                        
#         -ONRegresor: Searches for the optimum number of neighbours to
#                      predict the output of the data by plotting the error vs
#                      number of neighbours. The user select the number of
#                      neighbours after seeing the plot.
#                      
#     Coded by Eddy Martínez
#              20209153
#     MSc Robotics Engineering, Liverpool Hope University
# =============================================================================

def Classifier(X_train, Y_train, X_test, n_neighbors):
    from sklearn.neighbors import KNeighborsClassifier
    KNN_cl = KNeighborsClassifier(n_neighbors, weights='distance')
    KNN_cl.fit(X_train, Y_train)
    Y_pred = KNN_cl.predict(X_test)
    return Y_pred

def Regressor(X_train, Y_train, X_test, n_neighbors):
    from sklearn.neighbors import KNeighborsRegressor
    KNN_rg = KNeighborsRegressor(n_neighbors)
    KNN_rg.fit(X_train, Y_train)
    Y_pred = KNN_rg.predict(X_test)
    return Y_pred

def ONClassifier(X_train, X_test, Y_train, Y_test, neighbors):
    #To select the best N_Neighbours, we will plot the error vs.
    #the number of neighbours from 1 to 1/4 of the data points
    # quarter = int(len(Y_train)/4)+1
    # Error = [0]*(quarter-1)
    
    # for n_neighbors in range(1,quarter):
    #     Y_pred = Classifier(X_train, Y_train, X_test, n_neighbors)

    #     for i, j in zip(Y_test, Y_pred):
    #         if i!=j:
    #             Error[n_neighbors - 1] += 1
    #     n_neighbors += 50

    # import matplotlib.pyplot as plt
    # plt.plot(range(1,quarter),Error)
    # plt.xlabel('Number of Nearest Neighbours')
    # plt.ylabel('Number of incorrectly classified data points')
    # plt.show()

    # n_neighbors = int(input("Enter the number of nearest neighbours:\n"))
    
    Y_pred = Classifier(X_train, Y_train, X_test, neighbors)
    return Y_pred

def ONRegressor(X_train, X_test, Y_train, Y_test):
    # To select the best N_Neighbours, we will plot the error vs.
    # the number of neighbours from 1 to 1/4 of the data points
    # quarter = int(len(Y_train)/4)+1
    # Error = [0]*(quarter-1)

    # import numpy as np
    # for n_neighbors in range(1,quarter):
    #     Y_pred = Regressor(X_train, Y_train, X_test, n_neighbors)

    #     Error[n_neighbors - 1] = np.sqrt(np.sum((Y_test-Y_pred)**2)/len(Y_test))

    # import matplotlib.pyplot as plt
    # plt.plot(range(1,quarter),Error)
    # plt.xlabel('Number of Nearest Neighbours')
    # plt.ylabel('Standard Deviation of Predicted Data')
    # plt.show()
    
    # n_neighbors = int(input("Enter the number of nearest neighbours:\n"))
    
    Y_pred = Regressor(X_train, Y_train, X_test, n_neighbors=10)
    return Y_pred
