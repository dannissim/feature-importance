import enum

RANDOM_STATE = 15


class ModelTypes(enum.StrEnum):
    RANDOM_FOREST = 'random_forest'
    GRADIENT_BOOSTING = 'gradient_boosting'
    WAVELETS = 'wavelets'
    XGBOOST = 'xgboost'


class FeatureImportanceMethods(enum.StrEnum):
    RANDOM_FOREST = 'random_forest'
    GRADIENT_BOOSTING = 'gradient_boosting'
    WAVELETS = 'wavelets'
    XGBOOST = 'xgboost'
    CUSTOM1 = 'custom1'
    CUSTOM2 = 'custom2'
    CUSTOM3 = 'custom3'


FEATURE_IMPORTANCE_METHOD_TO_MODEL_MAPPING = {
    FeatureImportanceMethods.RANDOM_FOREST: ModelTypes.RANDOM_FOREST,
    FeatureImportanceMethods.GRADIENT_BOOSTING: ModelTypes.GRADIENT_BOOSTING,
    FeatureImportanceMethods.WAVELETS: ModelTypes.RANDOM_FOREST,
    FeatureImportanceMethods.XGBOOST: ModelTypes.XGBOOST,
    FeatureImportanceMethods.CUSTOM1: ModelTypes.RANDOM_FOREST,
    FeatureImportanceMethods.CUSTOM2: ModelTypes.RANDOM_FOREST,
    FeatureImportanceMethods.CUSTOM3: ModelTypes.RANDOM_FOREST
}
