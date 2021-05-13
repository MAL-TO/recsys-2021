import argparse
import os

from data.importer import import_data
from data.splitter import train_valid_test_split_bounds
from constants import ROOT_DIR
from util import pretty_evaluation, Stage, str2bool
from create_spark_context import create_spark_context

PATH_PREPROCESSED = os.path.join(ROOT_DIR, "../data/preprocessed")


def main(dataset_path, model_name, is_cluster):
    with Stage("Creating Spark context..."):
        create_spark_context(set_memory_conf=False)

    # graphframes module is only available after creating Spark context
    from preprocessor.features_store import FeatureStore

    with Stage("Initializing model..."):
        model = None
        if model_name == "native_xgboost_baseline":
            from model.native_xgboost_baseline import Model

            model = Model()
        if model_name == "h2o_xgboost_baseline":
            from model.h2o_xgboost_baseline import Model

            model = Model()

        assert model is not None, f"cannot find a model with name: {model_name}"
        enabled_features = model.enabled_features

    with Stage("Importing data..."):
        raw_data = import_data(dataset_path)

    with Stage("Assembling dataset..."):
        store = FeatureStore(PATH_PREPROCESSED, enabled_features, raw_data, is_cluster)
        features_union_df = store.get_dataset()

    with Stage("Split dataset"):
        # Train-valid-test split
        # TODO (manuele): .iloc with arrays is extremely slow on koalas.
        # Slicing works best. We should probably either:
        # i) pre-split data and load already split data in ks (best choice I think)
        # ii) sort (expensive) and then use slicing to get contiguous rows
        train_bounds, valid_bounds, test_bounds = train_valid_test_split_bounds(
            features_union_df
        )
        train_df = features_union_df.iloc[train_bounds["start"] : train_bounds["end"]]
        valid_df = features_union_df.iloc[valid_bounds["start"] : valid_bounds["end"]]
        test_df = features_union_df.iloc[test_bounds["start"] : test_bounds["end"]]

    with Stage("Fitting model"):
        hyperparams = {}
        model.fit(train_df, valid_df, hyperparams)

    with Stage("Evaluating model"):
        print(pretty_evaluation(model.evaluate(test_df)))


arg_parser = argparse.ArgumentParser(
    description="Process features, train a model, save it, and evaluate it"
)
arg_parser.add_argument("dataset_path", type=str, help="path to the full dataset")
arg_parser.add_argument("model_name", type=str, help="name of the model")
arg_parser.add_argument(
    "is_cluster", type=str2bool, help="are we running on the cluster"
)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    main(args.dataset_path, args.model_name, args.is_cluster)
