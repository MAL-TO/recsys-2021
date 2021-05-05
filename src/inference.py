import os
import gc

from pyspark import SparkConf, SparkContext
import databricks.koalas as ks

from constants import ROOT_DIR
from data.importer import import_data
from preprocessor.features_store import FeatureStore
from model.native_xgboost_baseline import Model
from util import Stage

PATH_PREPROCESSED = os.path.join(ROOT_DIR, "../data/preprocessed")
# TEST_DIR = "../data/raw/test"
TEST_DIR = "../test"


def main():
    with Stage("Initializing model..."):
        model = Model(include_targets=False)
        model.load_pretrained()

    part_files = [
        os.path.join(ROOT_DIR, TEST_DIR, f)
        for f in os.listdir(os.path.join(ROOT_DIR, TEST_DIR))
        if "part" in f
    ]

    with Stage("Creating Spark context..."):
        # https://spark.apache.org/docs/latest/configuration.html
        conf = SparkConf()
        conf.set("spark.driver.memory", "4g")
        conf.set("spark.driver.maxResultSize", "4g")

        conf.set("spark.executor.memory", "3g")

        conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        conf.set("spark.sql.execution.arrow.enabled", "true")
        conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")

        conf.setMaster("local[*]")
        conf.setAppName("Recsys-2021")

        SparkContext(conf=conf).setLogLevel("WARN")

        ks.set_option("compute.default_index_type", "distributed")

    for part_file in part_files:
        print(f"Processing '{part_file}'")

        with Stage("Importing data..."):
            raw_data = import_data(part_file, include_targets=False)

        with Stage("Assembling dataset..."):
            store = FeatureStore(
                PATH_PREPROCESSED,
                model.enabled_features,
                raw_data,
                is_cluster=False,
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
