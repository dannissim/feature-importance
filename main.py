import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np


def main():
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    feature_importance_result = our_feature_importance(rf_model)
    tree = rf_model.estimators_[0].tree_
    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

def our_feature_importance(rf_model):
    return np.sum((feature_importance_of_tree(tree.tree_) for tree in rf_model.estimators_), axis=0) / rf_model.n_estimators
def feature_importance_of_tree(tree):
    # feature_scores = [0] * tree.n_features
    feature_scores = np.zeros(shape=tree.n_features, dtype=np.float64)
    node_id_range = range(tree.node_count)
    node_depths = [0] * tree.node_count
    get_all_node_depths_recursive(tree, node_depths)
    for node_id in node_id_range:
        feature_used = tree.feature[node_id]
        if feature_used == -2:
            continue
        depth = node_depths[node_id]
        feature_scores[feature_used] += (1 / 2) ** depth
    return feature_scores


def get_all_node_depths_recursive(tree, node_depths, node_id=0, current_depth=1):
    node_depths[node_id] = current_depth
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    if left_child != -1:
        get_all_node_depths_recursive(tree, node_depths,  left_child, current_depth + 1)
    if right_child != -1:
        get_all_node_depths_recursive(tree, node_depths, right_child, current_depth + 1)
    return node_depths


if __name__ == "__main__":
    main()
