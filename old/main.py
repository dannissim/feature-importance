import typing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from random_forest import WaveletsForestRegressor
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.tree._tree import Tree

Model = typing.Union[RandomForestRegressor, WaveletsForestRegressor, GradientBoostingRegressor]

RANDOM_STATE = 15
RUN_IN_PARALLEL = -1
VERBOSE = 1
COLUMNS_AXIS = 1
REGRESSION_MODE = 'regression'

# Hyperparameters:
TEST_SIZE = 0.14
N_ESTIMATORS = 100
VARIABLE_IMPORTANCE_THRESHOLD = 0.8
MAX_DEPTH = None


def fit_model_to_data(model: Model, dataset: pd.DataFrame, target_column_name: str) -> typing.Tuple[Model, float]:
    y = dataset[target_column_name]
    X = dataset.drop(target_column_name, axis=COLUMNS_AXIS)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
    return mean_squared_error


def main():
    model = WaveletsForestRegressor(mode='regression',  trees=12, depth=None, seed=RANDOM_STATE,
                                    max_m_terms=10000, vi_threshold=0.001)
    model2 = RandomForestRegressor(n_estimators=5, max_depth=9, random_state=RANDOM_STATE, n_jobs=-1)

    df_diabetes = datasets.load_diabetes(as_frame=True).frame
    df_iris = datasets.load_iris(as_frame=True).frame
    df_breast_cancer = datasets.load_breast_cancer(as_frame=True).frame
    df_wine = datasets.load_wine(as_frame=True).frame
    df_digits = datasets.load_digits(as_frame=True).frame
    x = fit_model_to_data(model, df_wine, 'target')
    y = fit_model_to_data(model2, df_wine, 'target')
    z = 0


if __name__ == "__main__":
    print(main())
