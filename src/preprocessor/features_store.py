import json
import databricks.koalas as ks
from collections.abc import Mapping

# Example of custom features import
from preprocessor.targets import is_positive


class FeatureStore():
    """Handle feature configuration"""
    
    def __init__(self, path_preprocessed, enabled_features, raw_data):
        """Read configuration file (config.json)

        Args:
            preprocessed_path (str): materialized features files location
            enabled_features (list): list of the enabled features
            raw_data (ks.Dataframe): reference to raw data Dataframe instance
        """
        self.path_preprocessed = path_preprocessed
        self.enabled_features = enabled_features
        self.raw_data = raw_data # All features, in order to be able to extract    


    def extract_features(self):
        """[summary]

        Args:
            raw_data ([type]): [description]
            
        Returns:
            features_list: ...
        """
        
        feature_dict = {}  # dict of Series, each is a feature {feature_name: Series}
        
        for feature_name in self.enabled_features["custom"]:
            # TODO: define most suitable extension (csv, txt, ...)
            feature_path = os.path.join(self.path_preprocessed, feature_name + '.csv')

            # check if feature is already materialized in self.path_preprocessed
            if os.path.exists(feature_path):
                ks_feature = ks.read_csv(feature_path)

                if isinstance(extracted, Mapping):  # more than one feature extracted
                    feature_dict.update(ks_feature)
                else:
                    feature_dict[feature_name] = ks_feature

            else:
                print("### Extracting " + feature_name + "...")
                feature_extractor = globals()[feature_name]
                extracted = feature_extractor(raw_data=self.raw_data,
                                            features=feature_dict)
                
                if isinstance(extracted, Mapping):  # more than one feature extracted
                    feature_dict.update(extracted)
                else:
                    feature_dict[feature_name] = extracted

                # Store the new feature
                ks.to_csv(feature_path, )
                print("Feature added to " + feature_path)
                
        # assign feature names to series
        for k in feature_dict:
            feature_dict[k].name = k
        
        return feature_dict


    def get_dataset(self):
        """
        Returns:
            dataset (ks.DataFrame): the union of self.raw_data and the extracted features
        """
        # {'feature_name': Series, ...}
        feature_dict = self.extract_features()
        sliced_raw_data = self.raw_data.loc[:, self.enabled_features["default"]]

        # TODO: handle "custom" or "default"  empty, if it gives error
        features_dataset = ks.concat([sliced_raw_data] + list(feature_dict.values()), axis=1)

        return features_dataset