import os
import gc

from constants import ROOT_DIR, PATH_PREPROCESSED, PATH_AUXILIARIES
from data.importer import import_data
from model.h2o_xgboost_baseline import Model
from util import Stage
from create_spark_context import create_spark_context

TEST_DIR = "../data/raw/test"
# TEST_DIR = "../test"


def main():
    with Stage("Creating Spark context..."):
        create_spark_context()

    # graphframes module is only available after creating Spark context
    from preprocessor.features_store import FeatureStore

    with Stage("Initializing model..."):
        model = Model(include_targets=False)
        model.load_pretrained()

    part_files = [
        os.path.join(ROOT_DIR, TEST_DIR, f)
        for f in os.listdir(os.path.join(ROOT_DIR, TEST_DIR))
        if "part" in f
    ]

    for part_file in part_files:
        print(f"Processing '{part_file}'")

        with Stage("Importing data..."):
            raw_data = import_data(part_file, include_targets=False)

        with Stage("Assembling dataset..."):
            store = FeatureStore(
                PATH_PREPROCESSED,
                None,
                model.enabled_extractors,
                PATH_AUXILIARIES,
                None,
                model.enabled_auxiliaries,
                raw_data,
                is_cluster=False,
                is_inference=True,
            )
            features_union_df = store.get_dataset()

        with Stage("Computing predictions..."):
            predictions_df = model.predict(features_union_df).to_koalas(
                index_col=["tweet_id", "engaging_user_id"]
            )

        with Stage("Verifying length"):
            assert len(features_union_df) == len(
                predictions_df
            ), "features and predictions must have the same length"

        with Stage("Saving results.csv"):
            predictions_df.to_csv(
                path="results_folder",
                num_files=1,
                header=False,
                index_col=["tweet_id", "engaging_user_id"],
            )

        gc.collect()


if __name__ == "__main__":
    main()
