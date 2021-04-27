import databricks.koalas as ks
from collections.abc import Mapping


# Import features. Functions to extract features not in preprocessed folder
from preprocessor.targets.is_positive import is_positive

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
        feature_path = os.path.join(PATH_PREPROCESSED, feature_name + '.csv')

        # check if feature is already materialized in PATH_PREPROCESSED
        if os.path.exists(feature_path):
            ks_feature = ks.read_csv(feature_path, header=None)

            if isinstance(extracted, Mapping):  # more than one feature extracted
                feature_dict.update(ks_feature)
            else:
                feature_dict[feature_name] = ks_feature

        else:
            print("### Extracting " + feature_name + "...")
            feature_extractor = globals()[feature_name]
            extracted = feature_extractor(raw_data=raw_data,
                                        features=feature_dict)
            
            if isinstance(extracted, Mapping):  # more than one feature extracted
                feature_dict.update(extracted)
            else:
                feature_dict[feature_name] = extracted

            # Store the new feature
            ks.to_csv(feature_path, )
            print("Done. Feature has been added to " + os.path(PATH_PREPROCESSED))
            
    # assign feature names to series
    for k in feature_dict:
        feature_dict[k].name = k
    
    return feature_dict
