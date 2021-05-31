import argparse
import os

from data.importer import import_data
from data.splitter import train_valid_test_split_bounds
from constants import ROOT_DIR
from util import pretty_evaluation, Stage, str2bool
from create_spark_context import create_spark_context

PATH_PREPROCESSED = os.path.join(ROOT_DIR, "../data/preprocessed")


def main(dataset_path):
    is_cluster = False
    with Stage("Creating Spark context..."):
        create_spark_context(set_memory_conf=False)

    # graphframes module is only available after creating Spark context
    from preprocessor.features_store import FeatureStore
    
    enabled_features = [
            "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaging_user_follower_count",
            "engaging_user_following_count",
            "bert_test"
    ]

    with Stage("Importing data..."):
        raw_data = import_data(dataset_path)

    with Stage("Assembling dataset..."):
        store = FeatureStore(PATH_PREPROCESSED, enabled_features, raw_data, is_cluster)
        features_union_df = store.get_dataset()


arg_parser = argparse.ArgumentParser(
    description="Process features"
)
arg_parser.add_argument("dataset_path", type=str, help="path to the full dataset")

if __name__ == "__main__":
    args = arg_parser.parse_args()
    main(args.dataset_path)
