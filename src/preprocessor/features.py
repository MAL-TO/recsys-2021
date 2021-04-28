from collections.abc import Mapping

# Import features
from preprocessor.targets.is_positive import is_positive
from preprocessor.targets.binarize_timestamps import binarize_timestamps

PATH_PREPROCESSED = "../../../data/preprocessed"


def extract_features(raw_data, feature_config):
    """[summary]

    Args:
        raw_data ([type]): [description]
        feature_config ([type]): [description]

    Returns:
        features_list: ...
    """

    feature_dict = {}  # dict of Series, each is a feature {feature_name: Series}

    for feature_name in feature_config.get_enabled_custom_features_list():
        # TODO [feature store]: check if feature is already available in PATH_PREPROCESSED

        feature_extractor = globals()[feature_name]
        extracted = feature_extractor(raw_data=raw_data,
                                      features=feature_dict)

        if isinstance(extracted, Mapping):  # more than one feature extracted
            feature_dict.update(extracted)
        else:
            feature_dict[feature_name] = extracted

    # assign feature names to series
    for k in feature_dict:
        feature_dict[k].name = k

    return feature_dict
