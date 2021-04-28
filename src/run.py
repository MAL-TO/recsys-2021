import argparse
import os

import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession

from data.importer import import_data
from data.splitter import train_valid_test_split_bounds

from preprocessor.config import FeatureConfig
from preprocessor.features import extract_features

# TODO(Andrea): should this depend on a command line argument?
from model.native_xgboost_baseline import Model

from constants import ROOT_DIR

# Files and paths
RAW_DATA_INPUT_PATH = {
    "cluster_complete": "hdfs://BigDataHA/user/s277309/recsys_data/",
    "cluster_sample_one_hund": "hdfs://BigDataHA/user/s277309/recsys_data_sample/one_hund/",
    "cluster_sample_200k": "hdfs://BigDataHA/user/s277309/recsys_data_sample/local/sample200k",
    "local_sampled_small": os.path.join(ROOT_DIR, "../data/raw/sample200k"),
    "test": os.path.join(ROOT_DIR, "../test"),
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

    # TODO [feature store]: store non-available features

    # Train-valid-test split
    # TODO (manuele): .iloc with arrays is extremely slow on koalas.
    # Slicing works best. We should probably either:
    # i) pre-split data and load already split data in ks (best choice I think)
    # ii) sort (expensive) and then use slicing to get contiguous rows
    train_bounds, valid_bounds, test_bounds = train_valid_test_split_bounds(features_union_df)
    train_df = features_union_df.iloc[train_bounds['start']:train_bounds['end']]
    valid_df = features_union_df.iloc[valid_bounds['start']:valid_bounds['end']]
    test_df = features_union_df.iloc[test_bounds['start']:test_bounds['end']]

    print("Fitting model")
    hyperparams = {}
    model = Model()
    model.fit(train_df, valid_df, hyperparams)

    print("Evaluating model")
    print(model.evaluate(test_df))

if __name__ == "__main__":
    # run.py [data predefined path or custom path] [model name]

    dataset_name = "local_sampled_small"

    main(dataset_name)
