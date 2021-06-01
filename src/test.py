import os
import gc
import argparse
import numpy as np

from constants import ROOT_DIR
from data.importer import import_data
from model.native_xgboost_baseline import Model
from util import Stage, str2bool, rm_dir_contents
from create_spark_context import create_spark_context

PATH_PREPROCESSED = os.path.join(ROOT_DIR, "../data/preprocessed")
PATH_AUXILIARIES = os.path.join(ROOT_DIR, "../data/auxiliary")

DATA_DIR = "../data/raw/"

# List of Python dicts, each dict represents a fold with train and test set
# filenames inside `DATA_DIR`
DATA_PATHS = [
    {"train": "", "test": ""},
]


def main(is_cluster):
    # Initialize dict to store evaluation results
    results = {"train": [], "test": []}

    with Stage("Creating Spark context..."):
        create_spark_context()

    # graphframes module is only available after creating Spark context
    from preprocessor.features_store import FeatureStore

    for i in range(len(DATA_PATHS)):
        print(f"Dataset {i+1}/{len(DATA_PATHS)}")

        with Stage("Initializing model..."):
            model = Model()
            enabled_extractors = model.enabled_extractors
            enabled_auxiliaries = model.enabled_auxiliaries

        with Stage("Importing train dataset..."):
            train_path = DATA_PATHS[i]["train"]
            raw_train_data = import_data(os.path.join(DATA_DIR, train_path))

        with Stage("Cleaning feature store..."):
            rm_dir_contents(PATH_PREPROCESSED)
            rm_dir_contents(PATH_AUXILIARIES)

        with Stage("Assembling train dataset..."):
            store = FeatureStore(
                PATH_PREPROCESSED,
                enabled_extractors,
                PATH_AUXILIARIES,
                enabled_auxiliaries,
                raw_train_data,
                is_cluster,
                is_inference=False,
            )
            train_features_union_df = store.get_dataset()

        with Stage("Fitting model..."):
            hyperparams = {}
            model.fit(train_features_union_df, None, hyperparams)

        with Stage("Evaluating training dataset predictions..."):
            train_results = model.evaluate(train_features_union_df)

        with Stage("Importing test dataset..."):
            test_path = DATA_PATHS[i]["test"]
            raw_test_data = import_data(os.path.join(DATA_DIR, test_path))

        with Stage("Cleaning feature store..."):
            rm_dir_contents(PATH_PREPROCESSED)

        with Stage("Assembling test dataset..."):
            store = FeatureStore(
                PATH_PREPROCESSED,
                enabled_extractors,
                PATH_AUXILIARIES,
                enabled_auxiliaries,
                raw_test_data,
                is_cluster,
                is_inference=True,
            )
            test_features_union_df = store.get_dataset()

        with Stage("Evaluating test dataset predictions..."):
            test_results = model.evaluate(test_features_union_df)

        results["train"].append(train_results)
        results["test"].append(test_results)

        gc.collect()

    # Print results
    # TODO: Save (richer) results to a log file and maybe de-uglify this code
    for ds in results:
        print(f"### Results on {ds} set ###")

        res = {}
        for results_i in results[ds]:
            for metric in results_i:
                res.setdefault(metric, []).append(results_i[metric])

        all_AP = []
        all_RCE = []
        for metric in ["reply", "retweet", "retweet_with_comment", "like"]:
            vals_AP = np.array(res[metric + "_AP"])
            vals_RCE = np.array(res[metric + "_RCE"])
            all_AP.append(vals_AP)
            all_RCE.append(vals_RCE)

            print("------------------------------------")
            print(f"AP {metric}\t LB {(vals_AP.mean() - vals_AP.std()):+.4f} to UB {(vals_AP.mean() + vals_AP.std()):+.4f} (± 1σ)".expandtabs(30))
            print(f"RCE {metric}\t LB {(vals_RCE.mean() - vals_RCE.std()):+.4f} to UB {(vals_RCE.mean() + vals_RCE.std()):+.4f} (± 1σ)".expandtabs(30))

        all_AP = np.array(all_AP)
        all_RCE = np.array(all_RCE)
        print("------------------------------------")
        print(f"mAP\t LB {(all_AP.mean() - all_AP.std()):+.4f} to UB {(all_AP.mean() + all_AP.std()):+.4f} (± 1σ)".expandtabs(30))
        print(f"mRCE\t LB {(all_RCE.mean() - all_RCE.std()):+.4f} to UB {(all_RCE.mean() + all_RCE.std()):+.4f} (± 1σ)".expandtabs(30))
        print("------------------------------------")
        print()


arg_parser = argparse.ArgumentParser(
    description="Test a model"
)
arg_parser.add_argument(
    "is_cluster", type=str2bool, help="are we running on the cluster"
)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    main(args.is_cluster)
