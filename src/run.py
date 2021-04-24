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

from model.h2o_xgboost_baseline import H2OXGBoostBaseline

# Files and paths
RAW_DATA_INPUT_PATH = {
    "cluster_complete": "hdfs://BigDataHA/user/s277309/recsys_data/",
    "cluster_sampled": "hdfs://BigDataHA/user/s277309/recsys_data_sample/one_hund/",
    "local_sampled": "../data/raw/sample_0.0134_noid_notext_notimestamp.parquet",
    "local_sampled_small": "../data/raw/sample200k",
}

FEATURE_CONFIG_FILE = os.path.join("preprocessor", "config.json")


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
    
    hyperparams = {
        "ntrees" : 10,
        "max_depth" : 20,
        "learn_rate" : 0.1,
        "sample_rate" : 0.7,
        "col_sample_rate_per_tree" : 0.9,
        "min_rows" : 5,
        "seed": 42,
        "score_tree_interval": 100
    }
    
    print("Fitting model")
    xgboost = H2OXGBoostBaseline()
    xgboost.fit(train_df, valid_df, hyperparams)
    
    print("Evaluating model")
    # test trained model
    metrics = xgboost.evaluate(test_df,
                               save_to_logs=False)


if __name__ == "__main__":
    # run.py [data predefined path or custom path] [model name] 
    
    dataset_name = "local_sampled_small"
    
    main(dataset_name)
