import os
import gc
import argparse
import numpy as np
import h2o
from pysparkling import H2OContext

from constants import (
    PATH_PREPROCESSED,
    PATH_PREPROCESSED_CLUSTER,
    PATH_AUXILIARIES,
    PATH_AUXILIARIES_CLUSTER,
    MODEL_SEED,
    PATH_DATA,
    PATH_DATA_CLUSTER,
    FILENAMES_DATA,
)
from data.importer import import_data
from model.h2o_xgboost_pysparkling import Model
from util import Stage, str2bool, rm_dir_contents
from create_spark_context import create_spark_context

def main(is_cluster):
    # Initialize dict to store evaluation results
    results = {"train": [], "test": []}

    with Stage("Creating Spark context..."):
        sc = create_spark_context()

    with Stage("Creating H2O context..."):
        hc = H2OContext.getOrCreate()

    # graphframes module is only available after creating Spark context
    from preprocessor.features_store import FeatureStore

    if is_cluster:
        p_data = PATH_DATA_CLUSTER
    else:
        p_data = PATH_DATA

    for i in range(len(FILENAMES_DATA)):
        print(f"Dataset {i+1}/{len(FILENAMES_DATA)}")

        with Stage("Initializing model..."):
            model = Model(seed=MODEL_SEED)
            enabled_extractors = model.enabled_extractors
            enabled_auxiliaries = model.enabled_auxiliaries

        with Stage("Importing train dataset..."):
            fp_train = FILENAMES_DATA[i]["train"]
            raw_train_data = import_data(os.path.join(p_data, fp_train))

        with Stage("Cleaning feature store..."):
            rm_dir_contents(PATH_PREPROCESSED)
            rm_dir_contents(PATH_AUXILIARIES)

        with Stage("Assembling train dataset..."):
            store = FeatureStore(
                PATH_PREPROCESSED,
                PATH_PREPROCESSED_CLUSTER,
                enabled_extractors,
                PATH_AUXILIARIES,
                PATH_AUXILIARIES_CLUSTER,
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
            fp_test = FILENAMES_DATA[i]["test"]
            raw_test_data = import_data(os.path.join(p_data, fp_test))

        with Stage("Cleaning feature store..."):
            # Keep auxiliaries extracted from train set
            rm_dir_contents(PATH_PREPROCESSED)

        with Stage("Assembling test dataset..."):
            store = FeatureStore(
                PATH_PREPROCESSED,
                PATH_PREPROCESSED_CLUSTER,
                enabled_extractors,
                PATH_AUXILIARIES,
                PATH_AUXILIARIES_CLUSTER,
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

    # TODO: Save (richer) results to a log file and maybe de-uglify this part
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
            print(f"AP {metric}\t LB {(vals_AP.mean() - vals_AP.std()):+.4f} to UB {(vals_AP.mean() + vals_AP.std()):+.4f} (?? 1??)".expandtabs(30))
            print(f"RCE {metric}\t LB {(vals_RCE.mean() - vals_RCE.std()):+.4f} to UB {(vals_RCE.mean() + vals_RCE.std()):+.4f} (?? 1??)".expandtabs(30))

        all_AP = np.array(all_AP)
        all_RCE = np.array(all_RCE)
        print("------------------------------------")
        print(f"mAP\t LB {(all_AP.mean() - all_AP.std()):+.4f} to UB {(all_AP.mean() + all_AP.std()):+.4f} (?? 1??)".expandtabs(30))
        print(f"mRCE\t LB {(all_RCE.mean() - all_RCE.std()):+.4f} to UB {(all_RCE.mean() + all_RCE.std()):+.4f} (?? 1??)".expandtabs(30))
        print("------------------------------------")
        print()
    
    with Stage("Closing..."):
        h2o.cluster().shutdown(prompt=False)
        sc.stop()


arg_parser = argparse.ArgumentParser(
    description="Test a model"
)
arg_parser.add_argument(
    "is_cluster", type=str2bool, help="are we running on the cluster"
)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    main(args.is_cluster)
