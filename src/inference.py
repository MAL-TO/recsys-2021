import os

import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession

from constants import ROOT_DIR

from data.importer import import_data
from data.splitter import train_valid_test_split_bounds

from preprocessor.config import FeatureConfig
from preprocessor.features import extract_features

# TODO(Andrea): should this depend on a command line argument?
from model.native_xgboost_baseline import Model

# Files and paths
RAW_DATA_INPUT_PATH = {
    "cluster_complete": "hdfs://BigDataHA/user/s277309/recsys_data/",
    "cluster_sample_one_hund": "hdfs://BigDataHA/user/s277309/recsys_data_sample/one_hund/",
    "cluster_sample_200k": "hdfs://BigDataHA/user/s277309/recsys_data_sample/local/sample200k",
    "local_sampled": "../data/raw/sample_0.0134_noid_notext_notimestamp.parquet",
    "local_sampled_small": os.path.join(ROOT_DIR, "../data/raw/sample200k"),
}

FEATURE_CONFIG_FILE = os.path.join(ROOT_DIR, "preprocessor", "config.json")


def main(dataset_name):
    feature_config = FeatureConfig(FEATURE_CONFIG_FILE)

    print("Importing data")
    raw_data = import_data(dataset_name,
                           path=RAW_DATA_INPUT_PATH[dataset_name])

    # TODO [feature store]: only extract features that do not
    # already exist in data/preprocessed

    print("Extracting features")
    extracted_features = extract_features(raw_data, feature_config)

    # Drop raw_data features not enabled in config.json
    enabled_default_features = feature_config.get_enabled_default_features_list()
    raw_data = raw_data[enabled_default_features]

    # Merge all into a single dataframe with index
    features_union_df = ks.concat([raw_data] + list(extracted_features.values()), axis=1)

    print("Loading model")
    model = Model()
    model.load_pretrained()

    print('Computing predictions')
    predictions_df = model.predict(features_union_df)

    assert len(features_union_df) == len(predictions_df), \
        "features and predictions must have the same length"

    predictions_df['tweet_id'] = features_union_df['tweet_id'].to_numpy()
    predictions_df['user_id'] = features_union_df['engaging_user_id'].to_numpy()

    predictions_df[
        ['tweet_id', 'user_id', 'reply', 'retweet', 'retweet_with_comment', 'like']
    ].to_csv('results.csv', mode='a', header=False, index=False)

if __name__ == "__main__":
    # run.py [data predefined path or custom path] [model name]

    dataset_name = "local_sampled_small"

    main(dataset_name)
