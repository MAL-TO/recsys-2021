import os
import databricks.koalas as ks
from typing import Dict

from preprocessor.targets.binarize_timestamps import binarize_timestamps  # noqa: F401


class FeatureStore:
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
        # True if working on cluster, False if working on local machine
        self.is_cluster = is_cluster

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
                ks_feature = ks.read_csv(
                    feature_path, header=0, index_col=["tweet_id", "engaging_user_id"]
                )

                assert len(ks_feature) == len(self.raw_data)

                if isinstance(ks_feature, ks.DataFrame):
                    for column in ks_feature:
                        assert isinstance(ks_feature[column], ks.Series)
                        feature_dict[column] = ks_feature[column]
                elif isinstance(ks_feature, ks.Series):
                    feature_dict[feature_name] = ks_feature
                else:
                    raise TypeError(
                        f"ks_feature must be a Koalas DataFrame or Series, got {type(ks_feature)}"
                    )

            else:
                print("### Extracting " + feature_name + "...")
                feature_extractor = globals()[feature_name]
                extracted = feature_extractor(self.raw_data, feature_dict)

                if isinstance(extracted, dict):  # more than one feature extracted
                    for column in extracted:
                        assert isinstance(extracted[column], ks.Series)
                        feature_dict[column] = extracted[column]

                    # Store the new features
                    features_df = ks.concat(
                        list(extracted.values()), axis=1, join="inner"
                    )
                    assert len(features_df) == len(list(extracted.values())[0])
                    features_df.to_csv(
                        feature_path,
                        index_col=["tweet_id", "engaging_user_id"],
                        header=list(extracted.keys()),
                        num_files=(None if self.is_cluster else 1),
                    )
                elif isinstance(extracted, ks.Series):
                    feature_dict[feature_name] = extracted
                    extracted.to_csv(
                        feature_path,
                        index_col=["tweet_id", "engaging_user_id"],
                        header=[feature_name],
                        num_files=(None if self.is_cluster else 1),
                    )
                else:
                    raise TypeError(
                        f"extracted must be a Koalas DataFrame or Series, got {type(extracted)}"
                    )

                print("Feature added to " + feature_path)

        # Assign feature names to series
        for k in feature_dict:
            feature_dict[k].name = k

        return feature_dict

    def get_dataset(self):
        feature_dict = self.extract_features()
        sliced_raw_data: ks.DataFrame = self.raw_data[self.enabled_features["default"]]
        features_dataset = sliced_raw_data

        # NOTE: I suppose this could be done in one pass, with a multiple inner join.
        # IDK if it would be faster
        ks.set_option("compute.ops_on_diff_frames", True)
        for feature_name, feature_series in feature_dict.items():
            features_dataset = features_dataset.join(
                right=feature_series, on=["tweet_id", "engaging_user_id"], how="inner"
            )
        ks.set_option("compute.ops_on_diff_frames", False)

        # NOTE: the .sort_index is useful for having the rows always in the same
        # order. In this way the train_valid_test_split, which depends on the row's
        # order is the same regardless of how we joined. This allows for better
        # reproducibility and mental health.
        return features_dataset.sort_index()
