import json
import databricks.koalas as ks
from collections.abc import Mapping


class FeatureStore():
    """Handle feature configuration"""
    
    def __init__(self, preprocessed_path, enabled_features):
        """Read configuration file (config.json)

        Args:
            config_path (str): configuration file location
        """
        self.preprocessed_path = preprocessed_path
            
    def is_feature_enabled(self, feature_name):
        return {**self.config["default"], **self.config["custom"]}[feature_name]
    
    def get_enabled_features_list(self, type):
        features = [feat for feat in self.config[type] if self.is_feature_enabled(feat)]
        return features
    
    def get_enabled_custom_features_list(self):
        features = self.get_enabled_features_list("custom")
        return features

    def get_enabled_default_features_list(self):
        features = self.get_enabled_features_list("default")
        return features

    # Still based on config.json
    def extract_features(raw_data):
        """[summary]

        Args:
            raw_data ([type]): [description]
            
        Returns:
            features_list: ...
        """
        
        feature_dict = {}  # dict of Series, each is a feature {feature_name: Series}
        
        for feature_name in self.get_enabled_custom_features_list():
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