# =============================================================================
#     Splits the data in training and testing sets
#     then scales all features after the feature training set.
#                      
#     Coded by Eddy Martínez
#              20209153
#     MSc Robotics Engineering, Liverpool Hope University
# =============================================================================

def ScaleSplit(X, Y, size):
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = size, shuffle=True)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_train)
    X_train_sc = sc.transform(X_train)
    X_test_sc = sc.transform(X_test)

    return X_train_sc, X_test_sc, Y_train, Y_test