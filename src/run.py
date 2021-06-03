import argparse
import os

from data.importer import import_data
from constants import PATH_PREPROCESSED, PATH_PREPROCESSED_CLUSTER, PATH_AUXILIARIES, PATH_AUXILIARIES_CLUSTER, MODEL_SEED
from util import Stage, str2bool
from create_spark_context import create_spark_context


def main(dataset_path, model_name, is_cluster):
    with Stage("Creating Spark context..."):
        create_spark_context(set_memory_conf=False)

    # graphframes module is only available after creating Spark context
    from preprocessor.features_store import FeatureStore

    with Stage("Initializing model..."):
        model = None
        if model_name == "native_xgboost_baseline":
            from model.native_xgboost_baseline import Model

            model = Model(seed=MODEL_SEED)
        if model_name == "h2o_xgboost_baseline":
            from model.h2o_xgboost_baseline import Model

            model = Model(seed=MODEL_SEED)

        assert model is not None, f"cannot find a model with name: {model_name}"
        enabled_extractors = model.enabled_extractors
        enabled_auxiliaries = model.enabled_auxiliaries

    with Stage("Importing data..."):
        raw_data = import_data(dataset_path)

    with Stage("Assembling dataset..."):
        store = FeatureStore(
            PATH_PREPROCESSED,
            PATH_PREPROCESSED_CLUSTER,
            enabled_extractors,
            PATH_AUXILIARIES,
            PATH_AUXILIARIES_CLUSTER,
            enabled_auxiliaries,
            raw_data,
            is_cluster,
            is_inference=False,
        )
        features_union_df = store.get_dataset()

    with Stage("Fitting model"):
        hyperparams = {}
        model.fit(features_union_df, None, hyperparams)


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
