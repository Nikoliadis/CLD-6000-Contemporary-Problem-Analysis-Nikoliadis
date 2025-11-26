from sklearn.feature_selection import SelectKBest, mutual_info_classif

def select_features(X_train, y_train, X_test, k=10):
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X_train, y_train)

    cols = selector.get_support(indices=True)
    X_train_selected = X_train.iloc[:, cols]
    X_test_selected = X_test.iloc[:, cols]

    return X_train_selected, X_test_selected, cols
