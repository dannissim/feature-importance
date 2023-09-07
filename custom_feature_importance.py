import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree._tree import Tree

from built_in_models import fit_built_in_model
from consts import ModelTypes
from datasets import Dataset


def custom_method_1(dataset: Dataset):
    if not hasattr(dataset, 'frequencies_matrix'):
        dataset.frequencies_matrix = build_frequencies_matrix_of_forest(
            fit_built_in_model(ModelTypes.RANDOM_FOREST, dataset))
    frequencies_matrix = dataset.frequencies_matrix
    max_depth = frequencies_matrix.shape[1]
    # 1 / 2^n
    depth_weights = 1 / np.power(2, np.arange(0, max_depth))
    return np.dot(frequencies_matrix, depth_weights)


def custom_method_2(dataset: Dataset):
    if not hasattr(dataset, 'frequencies_matrix'):
        dataset.frequencies_matrix = build_frequencies_matrix_of_forest(
            fit_built_in_model(ModelTypes.RANDOM_FOREST, dataset))
    frequencies_matrix = dataset.frequencies_matrix
    max_depth = frequencies_matrix.shape[1]
    # 1 / n
    depth_weights = 1 / np.arange(1, max_depth + 1)
    return np.dot(frequencies_matrix, depth_weights)


def custom_method_3(dataset: Dataset):
    if not hasattr(dataset, 'frequencies_matrix'):
        dataset.frequencies_matrix = build_frequencies_matrix_of_forest(
            fit_built_in_model(ModelTypes.RANDOM_FOREST, dataset))
    frequencies_matrix = dataset.frequencies_matrix
    max_depth = frequencies_matrix.shape[1]
    depth_weights = 1 / np.power(np.arange(1, max_depth + 1), 2)
    return np.dot(frequencies_matrix, depth_weights)


def build_frequencies_matrix_of_forest(forest: RandomForestRegressor) -> np.ndarray:
    max_depth_of_forest = max(tree.tree_.max_depth for tree in forest.estimators_)
    sum_of_all_frequencies_matrices = np.zeros((forest.n_features_in_, max_depth_of_forest))
    for tree in forest.estimators_:
        sum_of_all_frequencies_matrices += build_frequencies_matrix_of_tree(
            tree.tree_, max_depth_of_forest)
    return sum_of_all_frequencies_matrices


def build_frequencies_matrix_of_tree(tree: Tree, max_depth_of_forest: int) -> np.ndarray:
    result = np.zeros((tree.n_features, max_depth_of_forest))
    node_id_range = range(tree.node_count)
    node_depths = [0] * tree.node_count
    get_all_node_depths(tree, node_depths)
    for node_id in node_id_range:
        feature_used = tree.feature[node_id]
        if feature_used == -2:
            continue
        depth = node_depths[node_id]
        result[feature_used][depth - 1] += 1
    return result


def get_all_node_depths(tree: Tree,
                        node_depths: np.array,
                        node_id: int = 0,
                        current_depth: int = 1) -> np.array:
    node_depths[node_id] = current_depth
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    if left_child != -1:
        get_all_node_depths(tree, node_depths, left_child, current_depth + 1)
    if right_child != -1:
        get_all_node_depths(tree, node_depths, right_child, current_depth + 1)
    return node_depths
