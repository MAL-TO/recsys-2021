import json
import os
import databricks.koalas as ks
from collections.abc import Mapping
from databricks.koalas.config import set_option, reset_option
from typing import Dict

# Example of custom features import
from preprocessor.targets.is_positive import is_positive
from preprocessor.targets.binarize_timestamps import binarize_timestamps

class FeatureStore():
    """Handle feature configuration"""

    def __init__(self, path_preprocessed, enabled_features, raw_data, is_cluster):
        """

        Args:
            preprocessed_path (str): materialized features files location
            enabled_features (list): list of the enabled features
            raw_data (ks.Dataframe): reference to raw data Dataframe instance
        """
        self.path_preprocessed = path_preprocessed
        self.raw_data = raw_data
        self.is_cluster = is_cluster # True if working on cluster, False if working on local machine

        self.enabled_features = {"default": [], "custom": []}
        for feature in enabled_features:
            if feature in self.raw_data.columns:
                self.enabled_features["default"].append(feature)
            else:
                self.enabled_features["custom"].append(feature)


    def extract_features(self):
        feature_dict: Dict[str, ks.Series] = {}

        for feature_name in self.enabled_features["custom"]:
            feature_path = os.path.join(self.path_preprocessed, feature_name)

            # If feature already materialized
            if os.path.exists(feature_path):
                print("### Reading cached " + feature_name + "...")
                ks_feature = ks.read_csv(feature_path, header = 0, index_col = 'sorting_index')

                assert len(ks_feature) == len(self.raw_data)

                # Handles different partitions mantaining the order
                if self.is_cluster:
                    ks_feature.sort_index(inplace=True)

                # Drop 'sorting_index' column
                ks_feature.reset_index(drop=True, inplace=True)

                if isinstance(ks_feature, ks.DataFrame):
                    for column in ks_feature:
                        assert isinstance(ks_feature[column], ks.Series)
                        feature_dict[column] = ks_feature[column]
                elif isinstance(ks_feature, ks.Series):
                    feature_dict[feature_name] = ks_feature
                else:
                    raise TypeError(f"ks_feature must be a Koalas DataFrame or Series, got {type(ks_feature)}")

            else:
                print("### Extracting " + feature_name + "...")
                feature_extractor = globals()[feature_name]
                extracted = feature_extractor(self.raw_data, feature_dict)

                if isinstance(extracted, dict):  # more than one feature extracted
                    for column in extracted:
                        assert isinstance(extracted[column], ks.Series)
                        feature_dict[column] = extracted[column]

                    # Store the new features
                    # TODO(Francesco): ks.concat is slow and not adviced.
                    # ks.DataFrame(extraced) does not work with koalas, only with pandas - to_pandas() adviced only for small dataframes
                    features_df = ks.concat(list(extracted.values()), axis=1, join = 'inner')
                    if self.is_cluster:
                        features_df.to_csv(feature_path, index_col = ['sorting_index'], header = list(extracted.keys()))
                    else:
                        features_df.to_csv(feature_path, index_col = ['sorting_index'], header = list(extracted.keys()), num_files=1)

                elif isinstance(extracted, ks.Series):
                    feature_dict[feature_name] = extracted

                    # Store the new feature
                    if self.is_cluster:
                        extracted.to_csv(feature_path, index_col = ['sorting_index'], header = [feature_name])
                    else:
                        extracted.to_csv(feature_path, index_col = ['sorting_index'], header = [feature_name], num_files=1)
                else:
                    raise TypeError(f"extracted must be a Koalas DataFrame or Series, got {type(extracted)}")

                print("Feature added to " + feature_path)

        # Assign feature names to series
        for k in feature_dict:
            feature_dict[k].name = k

        return feature_dict

    # TODO(Francesco): when working on distributed environment, koalas.read_csv() loses the order of the row.
    # Option1: keep the index and use .join with index as key
    # Option2: keep the index, sort by index, use .concat()
    def get_dataset(self):
        feature_dict = self.extract_features()
        sliced_raw_data = self.raw_data[self.enabled_features["default"]].spark.local_checkpoint()

        features_dataset = sliced_raw_data.cache()

        ks.set_option('compute.ops_on_diff_frames', True)
        for feature_name, feature_series in feature_dict.items():
            assert len(sliced_raw_data) == len(feature_series)
            features_dataset[feature_name] = feature_series
        ks.set_option('compute.ops_on_diff_frames', False)

        assert len(features_dataset) == len(sliced_raw_data)
        for _, feature in feature_dict.items():
            assert len(features_dataset) == len(feature)

        return features_dataset
