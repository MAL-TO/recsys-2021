import json
import os
import databricks.koalas as ks
from collections.abc import Mapping
from databricks.koalas.config import set_option, reset_option

# Example of custom features import
from preprocessor.targets.is_positive import is_positive


class FeatureStore():
    """Handle feature configuration"""
    
    def __init__(self, path_preprocessed, enabled_features, raw_data, is_cluster):
        """Read configuration file (config.json)

        Args:
            preprocessed_path (str): materialized features files location
            enabled_features (list): list of the enabled features
            raw_data (ks.Dataframe): reference to raw data Dataframe instance
        """
        self.path_preprocessed = path_preprocessed
        self.enabled_features = enabled_features
        self.raw_data = raw_data # All features, in order to be able to extract  
        self.is_cluster = is_cluster # True if working on cluster, False if working on local machine  


    def extract_features(self):
        """[summary]

        Args:
            raw_data ([type]): [description]
            
        Returns:
            features_list: ...
        """
        
        feature_dict = {}  # dict of Series, each is a feature {feature_name: Series}
        
        for feature_name in self.enabled_features["custom"]:
            # TODO(Francesco): define most suitable extension (csv, txt, ...)
            feature_path = os.path.join(self.path_preprocessed, feature_name + '.csv')

            # If feature already materialized. ERROR HERE
            if os.path.exists(feature_path):
                ks_feature = ks.read_csv(feature_path, header = 0, index_col = 'sorting_index', names = feature_name)

                # Handles different partitions
                if self.is_cluster:
                    ks_feature.sort_index(inplace=True)

                if isinstance(ks_feature, Mapping):  # more than one feature extracted
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
                if self.is_cluster:
                    extracted.to_csv(feature_path, header=True, index_col = ['sorting_index'])
                else:
                    extracted.to_csv(feature_path, header=True, index_col = ['sorting_index'], num_files=1)

                print("Feature added to " + feature_path)
                
        # Assign feature names to series
        for k in feature_dict:
            feature_dict[k].name = k
        
        return feature_dict

    # TODO(Francesco): when working on distributed environment, koalas.read_csv() loses the order of the row.
    # Option1: keep the index and use .join with index as key
    # Option2: keep the index, sort by index, use .concat()
    def get_dataset(self):
        """
        Returns:
            dataset (ks.DataFrame): the union of self.raw_data and the extracted features
        """ 
        # Explicitly allow join on different dataframes. Please refer to https://docs.databricks.com/_static/notebooks/pandas-to-koalas-in-10-minutes.html
        set_option("compute.ops_on_diff_frames", True)

        # {'feature_name': Series, ...}
        feature_dict = self.extract_features()
        sliced_raw_data = self.raw_data.loc[:, self.enabled_features["default"]]
        features_dataset = ks.concat([sliced_raw_data] + list(feature_dict.values()), axis=1)

        # Reset to default to avoid potential expensive operation in the future
        reset_option("compute.ops_on_diff_frames")

        return features_dataset