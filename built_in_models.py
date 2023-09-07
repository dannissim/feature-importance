import typing

import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from consts import RANDOM_STATE, ModelTypes
from datasets import Dataset, get_partitioned_dataset
from random_forest import WaveletsForestRegressor

N_ESTIMATORS = 100
VERBOSE = 0
RUN_PARALLEL = -1

MODELS_FACTORY = {
    ModelTypes.RANDOM_FOREST:
    lambda: RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=RUN_PARALLEL, verbose=VERBOSE),
    ModelTypes.GRADIENT_BOOSTING:
    lambda: GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, verbose=VERBOSE),
    ModelTypes.WAVELETS:
    lambda: WaveletsForestRegressor(
        mode='regression', trees=N_ESTIMATORS, seed=RANDOM_STATE, depth=None, vi_threshold=0.8),
    ModelTypes.XGBOOST:
    lambda: xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS, n_jobs=RUN_PARALLEL, random_state=RANDOM_STATE)
}


def fit_built_in_model(
    model_type: ModelTypes, dataset: Dataset
) -> typing.Union[RandomForestRegressor, WaveletsForestRegressor, GradientBoostingRegressor,
                  xgb.XGBRegressor]:
    model = MODELS_FACTORY[model_type]()
    partitioned_dataset = get_partitioned_dataset(dataset.frame, dataset.target_column_name)
    model.fit(partitioned_dataset.X_train, partitioned_dataset.y_train)
    return model
