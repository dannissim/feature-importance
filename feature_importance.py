import itertools

import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

import custom_feature_importance
from built_in_models import MODELS_FACTORY, fit_built_in_model
from consts import (FEATURE_IMPORTANCE_METHOD_TO_MODEL_MAPPING,
                    FeatureImportanceMethods, ModelTypes)
from datasets import Dataset, get_datasets, get_partitioned_dataset

ROUNDING_PRECISION = 3
COMPRESSION_RATIOS = (0.2, 0.4, 0.6, 0.8, 1)


def main():
    feature_importances_df = pd.DataFrame({
        'dataset': [],
        'feature_importance_method': [],
        'feature_importance_results': []
    })
    mse_after_compression_df = pd.DataFrame({
        'dataset': [],
        'amount_of_features': [],
        'feature_importance_method': [],
        'model': [],
        'mse': []
    })
    datasets = get_datasets()
    progress_bar = tqdm(total=len(datasets) * len(FeatureImportanceMethods) *
                        len(COMPRESSION_RATIOS),
                        desc="Overall Progress")
    for dataset in datasets:
        for feature_importance_method in FeatureImportanceMethods:
            feature_importance_result = FEATURE_IMPORTANCE_METHOD_TO_FUNCTION_MAPPING[
                feature_importance_method](dataset)
            add_feature_importance_to_results(feature_importances_df, feature_importance_method,
                                              dataset.name, feature_importance_result)
            model_type = FEATURE_IMPORTANCE_METHOD_TO_MODEL_MAPPING[feature_importance_method]
            for compression_ratio in COMPRESSION_RATIOS:
                amount_of_features_after_compression = max(1, int(compression_ratio *
                                                           len(feature_importance_result)))
                mse_of_compressed_model_result = mse_of_compressed_model(
                    model_type, dataset, feature_importance_result,
                    amount_of_features_after_compression)
                add_mse_after_compression_to_results(mse_after_compression_df, dataset.name,
                                                     amount_of_features_after_compression,
                                                     feature_importance_method, model_type,
                                                     mse_of_compressed_model_result)
                progress_bar.update()
    return feature_importances_df, mse_after_compression_df


def random_forest_feature_importance(dataset: Dataset) -> np.array:
    return fit_built_in_model(ModelTypes.RANDOM_FOREST, dataset).feature_importances_


def wavelets_feature_importance(dataset: Dataset) -> np.array:
    return fit_built_in_model(ModelTypes.WAVELETS, dataset).feature_importances_


def gradient_boosting_feature_importance(dataset: Dataset) -> np.array:
    return fit_built_in_model(ModelTypes.GRADIENT_BOOSTING, dataset).feature_importances_


def xgboost_feature_importance(dataset: Dataset) -> np.array:
    return fit_built_in_model(ModelTypes.XGBOOST, dataset).feature_importances_


FEATURE_IMPORTANCE_METHOD_TO_FUNCTION_MAPPING = {
    FeatureImportanceMethods.RANDOM_FOREST: random_forest_feature_importance,
    FeatureImportanceMethods.WAVELETS: wavelets_feature_importance,
    FeatureImportanceMethods.GRADIENT_BOOSTING: gradient_boosting_feature_importance,
    FeatureImportanceMethods.XGBOOST: xgboost_feature_importance,
    FeatureImportanceMethods.CUSTOM1: custom_feature_importance.custom_method_1,
    FeatureImportanceMethods.CUSTOM2: custom_feature_importance.custom_method_2,
    FeatureImportanceMethods.CUSTOM3: custom_feature_importance.custom_method_3
}


def mse_of_compressed_model(model_type: ModelTypes, dataset: Dataset, feature_importance: np.array,
                            amount_of_features_after_compression: int) -> float:
    compressed_model = MODELS_FACTORY[model_type]()
    selected_feature_indices = list(
        itertools.islice(reversed(np.argsort(feature_importance)),
                         amount_of_features_after_compression))
    selected_column_indices = selected_feature_indices + [
        dataset.frame.columns.get_loc(dataset.target_column_name)
    ]
    trimmed_frame = dataset.frame.iloc[:, selected_column_indices]
    partitioned_dataset = get_partitioned_dataset(trimmed_frame, dataset.target_column_name)
    compressed_model.fit(partitioned_dataset.X_train, partitioned_dataset.y_train)
    y_pred_of_compressed_model = compressed_model.predict(partitioned_dataset.X_test)
    return round(metrics.mean_squared_error(partitioned_dataset.y_test, y_pred_of_compressed_model),
                 ROUNDING_PRECISION)


def add_feature_importance_to_results(feature_importances_df, feature_importance_method,
                                      dataset_name, feature_importance_result):
    feature_importance_row = dict()
    feature_importance_row['feature_importance_method'] = feature_importance_method
    feature_importance_row['dataset'] = dataset_name
    feature_importance_row['feature_importance_results'] = list(
        reversed(np.argsort(feature_importance_result)))
    feature_importances_df.loc[feature_importances_df.shape[0]] = feature_importance_row


def add_mse_after_compression_to_results(mse_after_compression_df, dataset_name, amount_of_features,
                                         feature_importance_method, model_type,
                                         mse_of_compressed_model_result):
    mse_after_compression_row = dict()
    mse_after_compression_row['dataset'] = dataset_name
    mse_after_compression_row['amount_of_features'] = amount_of_features
    mse_after_compression_row['feature_importance_method'] = feature_importance_method
    mse_after_compression_row['model'] = model_type
    mse_after_compression_row['mse'] = mse_of_compressed_model_result
    mse_after_compression_df.loc[mse_after_compression_df.shape[0]] = mse_after_compression_row


if __name__ == '__main__':
    main()
