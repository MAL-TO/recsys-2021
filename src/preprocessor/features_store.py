import os
import h2o
import databricks.koalas as ks
from typing import Dict

from preprocessor.targets.binarize_timestamps import binarize_timestamps  # noqa: F401
from preprocessor.time.hour_of_day import hour_of_day
from preprocessor.tweet.word_count import word_count
from preprocessor.encoding.te_language_hour import te_language_hour
from preprocessor.encoding.te_tweet_type_word_count import te_tweet_type_word_count
from preprocessor.encoding.te_user_lang import te_user_lang

from pysparkling import H2OContext
hc = H2OContext.getOrCreate()
index_cols = ['tweet_id', 'engaging_user_id']

class FeatureStore:
    """Handle feature configuration"""

    def __init__(
        self,
        path_preprocessed,
        path_preprocessed_cluster,
        enabled_extractors,
        path_auxiliaries,
        path_auxiliaries_cluster,
        enabled_auxiliaries,
        raw_data,
        is_cluster,
        is_inference,
    ):
        """

        Args:
            path_preprocessed (str): materialized features files location
            enabled_extractors (list): list of the enabled feature extractors
            path_auxiliaries (str): materialized auxiliary data location
            enabled_auxiliaries (list): list of enabled auxiliary sources
            raw_data (ks.Dataframe): reference to raw data Dataframe instance
            is_cluster (bool): True if working on cluster, False if working on
                local machine
            is_inference (bool): True if inference time, False if training time
        """
        self.path_preprocessed = path_preprocessed
        self.path_auxiliaries = path_auxiliaries
        self.raw_data = raw_data
        self.is_cluster = is_cluster
        self.is_inference = is_inference

        if self.is_cluster:
            self.path_preprocessed_rw = path_preprocessed_cluster
            self.path_auxiliaries_rw = path_auxiliaries_cluster
        else:
            self.path_preprocessed_rw = self.path_preprocessed
            self.path_auxiliaries_rw = self.path_auxiliaries

        self.enabled_auxiliaries = enabled_auxiliaries

        self.enabled_features = {"default": [], "custom": []}
        for feature in enabled_extractors:
            if feature in self.raw_data.columns:
                self.enabled_features["default"].append(feature)
            else:
                self.enabled_features["custom"].append(feature)

    def extract_auxiliaries(self):
        """
        Rationale: some features require additional data to be materialized.
        For instance, social network-related features require a graph
        representation of engagements, i.e., an additional DataFrame besides
        raw data and previously extracted features.

        Auxiliary data does not necessarily follow 1:1 correspondence between
        its rows and raw data rows.

        The same auxiliary data source may be exploited to extract multiple
        features.

        It is possible to use auxiliary data sources, pre-built on the training
        set, also at inference time, given that the auxiliary data source is not
        excessively large (still need to define what "too large" means, though
        we have ~20GB at our disposal on the test VM).
        """

        auxiliary_dict: Dict[str, ks.DataFrame] = {}

        for auxiliary_name in self.enabled_auxiliaries:
            auxiliary_path = os.path.join(self.path_auxiliaries, auxiliary_name)
            auxiliary_path_rw = os.path.join(self.path_auxiliaries_rw, auxiliary_name)

            # If auxiliary data is already materialized
            if os.path.exists(auxiliary_path):
                print("### Reading cached auxiliary data " + auxiliary_name + "...")
                auxiliary_list = self.get_subdir_list(auxiliary_path)

                for key in auxiliary_list:
                    ks_auxiliary = ks.read_csv(
                        os.path.join(auxiliary_path_rw, key), header=0
                    )
                    if isinstance(ks_auxiliary, ks.DataFrame):
                        auxiliary_dict[key] = ks_auxiliary
                    else:
                        raise TypeError(
                            f"ks_auxiliary must be a Koalas DataFrame, got {type(ks_auxiliary)}"
                        )

                # If inference time, and auxiliary data exists, it must be
                # extended with test data
                if self.is_inference:
                    print(
                        "### Integrating auxiliary data with test set "
                        + auxiliary_name
                        + "..."
                    )
                    auxiliary_extractor = globals()[auxiliary_name]
                    auxiliary_extracted = auxiliary_extractor(
                        self.raw_data, auxiliary_train=auxiliary_dict
                    )

                    if isinstance(auxiliary_extracted, dict):
                        for key in auxiliary_extracted:
                            assert isinstance(auxiliary_extracted[key], ks.DataFrame)
                            auxiliary_dict[key] = auxiliary_extracted[key]
                    else:
                        raise TypeError(
                            f"auxiliary_extracted must be a dict, got {type(auxiliary_extracted)}"
                        )

            else:
                print("### Extracting auxiliary data " + auxiliary_name + "...")
                auxiliary_extractor = globals()[auxiliary_name]
                auxiliary_extracted = auxiliary_extractor(self.raw_data)

                os.mkdir(auxiliary_path)
                if isinstance(auxiliary_extracted, dict):
                    for key in auxiliary_extracted:
                        assert isinstance(auxiliary_extracted[key], ks.DataFrame)
                        auxiliary_dict[key] = auxiliary_extracted[key]

                        auxiliary_extracted_path = os.path.join(auxiliary_path_rw, key)

                        # Store the current auxiliary dataframe
                        auxiliary_extracted[key].to_csv(
                            auxiliary_extracted_path,
                            index_col=None,  # TODO (Manuele) how to handle index_col?
                            header=list(auxiliary_extracted[key].columns),
                            num_files=1,
                        )
                else:
                    raise TypeError(
                        f"auxiliary_extracted must be a dict, got {type(auxiliary_extracted)}"
                    )

                print("Auxiliary data added to " + auxiliary_path)

        return auxiliary_dict

    def extract_features(self):
        auxiliary_dict = self.extract_auxiliaries()

        feature_dict: Dict[str, ks.Series] = {}

        for feature_name in self.enabled_features["custom"]:
            feature_path = os.path.join(self.path_preprocessed, feature_name)
            feature_path_rw = os.path.join(self.path_preprocessed_rw, feature_name)

            # If feature already materialized
            if os.path.exists(feature_path):
                print("### Reading cached " + feature_name + "...")
                ks_feature = ks.read_csv(
                    feature_path_rw,
                    header=0,
                    index_col=["tweet_id", "engaging_user_id"],
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
                extracted = feature_extractor(
                    self.raw_data, feature_dict, auxiliary_dict, self.is_inference
                )
                
                if isinstance(extracted, dict):  # more than one feature extracted
                    print("More than one feature")
                    
                    # Convert H2OFrames in ks.Series
                    for key, new_col in extracted.items():
                        if isinstance(new_col, h2o.H2OFrame):
                            # 1. Broken version: H2O -> spark -> koalas
                            new_col_spark = hc.asSparkFrame(new_col)
                            new_col_koalas = ks.DataFrame(new_col_spark).set_index(index_cols).squeeze()

                            # 2. Working version: H2O -> pandas -> koalas
#                             new_col_pandas =new_col.as_data_frame()
#                             new_col_koalas = ks.DataFrame(new_col_pandas).set_index(index_cols).squeeze()
                            new_col_koalas.name = key
                            extracted[key] = new_col_koalas
                            
                            print(extracted[key].head())
                
                    for column in extracted:
                        assert isinstance(extracted[column], ks.Series)
                        feature_dict[column] = extracted[column]

                    # Store the new features
                    print("Concat of multiple features")
                    features_df = ks.concat(
                        list(extracted.values()), axis=1, join="inner"
                    )
                    
                    assert len(features_df) == len(list(extracted.values())[0])
                    
                    print("to csv of multiple features")
                    features_df.to_csv(
                        feature_path_rw,
                        index_col=["tweet_id", "engaging_user_id"],
                        header=list(extracted.keys()),
                        num_files=1,
                    )
                elif isinstance(extracted, ks.Series):
                    feature_dict[feature_name] = extracted
                    extracted.to_csv(
                        feature_path_rw,
                        index_col=["tweet_id", "engaging_user_id"],
                        header=[feature_name],
                        num_files=1,
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
        ks.set_option("compute.ops_on_diff_frames", True)

        feature_dict = self.extract_features()
        sliced_raw_data: ks.DataFrame = self.raw_data[self.enabled_features["default"]]
        features_dataset = sliced_raw_data

        # NOTE: I suppose this could be done in one pass, with a multiple inner join.
        # IDK if it would be faster
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

    @staticmethod
    def get_subdir_list(path):
        subdirs = []
        for root, dirs, _ in os.walk(path):
            for dir_ in dirs:
                subdirs.append(dir_)
            break

        return subdirs
