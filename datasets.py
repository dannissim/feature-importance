from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split

from consts import RANDOM_STATE

COLUMNS_AXIS = 1
TEST_SIZE = 0.2


@dataclass
class PartitionedDataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame


@dataclass
class Dataset:
    frame: pd.DataFrame
    target_column_name: str
    name: str


def get_partitioned_dataset(frame: pd.DataFrame, target_column_name: str) -> PartitionedDataset:
    y = frame[target_column_name]
    X = frame.drop(target_column_name, axis=COLUMNS_AXIS)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)
    return PartitionedDataset(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def get_datasets():
    df_diabetes = Dataset(frame=sklearn.datasets.load_diabetes(as_frame=True).frame,
                          target_column_name='target',
                          name='diabetes')
    df_iris = Dataset(frame=sklearn.datasets.load_iris(as_frame=True).frame,
                      target_column_name='target',
                      name='iris')
    df_breast_cancer = Dataset(frame=sklearn.datasets.load_breast_cancer(as_frame=True).frame,
                               target_column_name='target',
                               name='breast_cancer')
    df_wine = Dataset(frame=sklearn.datasets.load_wine(as_frame=True).frame,
                      target_column_name='target',
                      name='wine')
    df_housing = Dataset(frame=sklearn.datasets.fetch_california_housing(as_frame=True).frame,
                         target_column_name='MedHouseVal',
                         name='housing')
    df_covtype = Dataset(frame=sklearn.datasets.fetch_covtype(as_frame=True).frame,
                         target_column_name='Cover_Type',
                         name='covtype')
    df_digits = Dataset(frame=sklearn.datasets.load_digits(as_frame=True).frame,
                        target_column_name='target',
                        name='digits')
    df_housing_with_noise = Dataset(frame=_add_noise(df_housing.frame),
                                    target_column_name='MedHouseVal',
                                    name='housing_with_noise')
    datasets_to_check = [
        df_diabetes, df_iris, df_breast_cancer, df_wine, df_housing, df_covtype, df_digits,
        df_housing_with_noise
    ]
    datasets_to_check_test = [df_wine]
    return datasets_to_check


def _add_noise(data_set: pd.DataFrame) -> pd.DataFrame:
    std_dev = data_set.std()
    num_rows, num_cols = data_set.shape
    num_elements = num_rows * num_cols
    num_to_change = int(num_elements * 0.1)
    row_indices = np.random.randint(0, num_rows, num_to_change)
    col_indices = np.random.randint(0, num_cols, num_to_change)
    noise = np.random.normal(0, std_dev[col_indices] * 0.5, num_to_change)
    for i in range(num_to_change):
        row = row_indices[i]
        col = data_set.columns[col_indices[i]]
        data_set.at[row, col] += noise[i]
    return data_set
