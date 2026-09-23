"""Exercise feature construction followed by the separate final validation."""

from entso_e_pipeline.features import engineering
from entso_e_pipeline.modeling.schema import split_target, validate_features


def validated_transform(*args, **kwargs):
    features = engineering.transform(*args, **kwargs)
    validate_features(features)
    return features


def validated_training(*args, **kwargs):
    result = engineering.build_labeled_features(*args, **kwargs)
    split_target(result[0])
    return result
