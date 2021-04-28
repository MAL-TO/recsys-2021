import argparse
import os

import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession

from data.importer import import_data
from data.splitter import train_valid_test_split_bounds

from preprocessor.features_store import FeatureStore

# TODO(Andrea): should this depend on a command line argument?
from model.native_xgboost_baseline import Model

# Files and paths
RAW_DATA_INPUT_PATH = {
    "cluster_complete": "hdfs://BigDataHA/user/s277309/recsys_data/",
    "cluster_sample_one_hund": "hdfs://BigDataHA/user/s277309/recsys_data_sample/one_hund/",
    "cluster_sample_200k": "hdfs://BigDataHA/user/s277309/recsys_data_sample/local/sample200k",
    "local_sampled": "../data/raw/sample_0.0134_noid_notext_notimestamp.parquet",
    "local_sampled_small": "../data/raw/sample200k",
    "local_custom_sample": "./data/raw/sample200.csv"
}

PATH_PREPROCESSED = 'data/preprocessed'
CLUSTER = False # TODO(Francesco) Command line argument - handles number of partitions in ks.to_csv


def main(dataset_name):

    print("Initializing model...", end = " ")
    model = Model()
    enabled_features = model.enabled_features
    print("Done")
    
    print("Importing data...", end=" ")
    raw_data = import_data(path=RAW_DATA_INPUT_PATH[dataset_name])  
    print("Done")

    print("Assembling dataset...")
    store = FeatureStore(PATH_PREPROCESSED, enabled_features, raw_data, CLUSTER)
    features_union_df = store.get_dataset()
    print("Dataset ready")

    
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
    model.fit(train_df, valid_df, hyperparams)
    
    print("Evaluating model")
    print(model.evaluate(test_df))

if __name__ == "__main__":
    # run.py [data predefined path or custom path] [model name] 
    
    dataset_name = "local_custom_sample"
    
    main(dataset_name)
