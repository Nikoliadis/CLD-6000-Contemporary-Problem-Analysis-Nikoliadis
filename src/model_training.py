# model_training.py

from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree model according to the Design Document.
    """
    model = DecisionTreeClassifier(
        max_depth=6,
        criterion='gini',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
