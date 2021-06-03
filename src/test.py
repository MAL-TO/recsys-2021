# Script for testing new features

import xgboost as xgb
# import matplotlib.pyplot as plt
# import seaborn as sns
import os
import numpy as np
os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1'
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

from data.importer import import_data
from util import pretty_evaluation, Stage
from cross_validate import cross_validate, cv_metrics
# from metrics import compute_score, just_mAP_mRCE, pretty_evaluation
from preprocessor.features_store import FeatureStore


# Arguments
DATASET_PATH = "../data/raw/time_sample200k.parquet"
PATH_PREPROCESSED = "../data/preprocessed"
PATH_AUXILIARIES = "../data/auxiliary"
IS_CLUSTER = False
ENABLED_FEATURES = [
    "engaged_with_user_follower_count",
    "engaged_with_user_following_count",
    "engaging_user_follower_count",
    "engaging_user_following_count",
    "hour_of_day",
    "te_language_hour",
    "word_count",
    "user_activity",

    "binarize_timestamps"
]

ENABLED_AUXILIARY = []


def main():
    with Stage("Importing data..."):
        raw_data = import_data(DATASET_PATH)
        
    with Stage("Assembling dataset..."):
        store = FeatureStore(
            PATH_PREPROCESSED,
            ENABLED_FEATURES,
            PATH_AUXILIARIES,
            ENABLED_AUXILIARY,
            raw_data,
            is_cluster=IS_CLUSTER,
            is_inference=False,
        )
        features_union_df = store.get_dataset()

    params = {
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'subsample': 0.7,
        'min_child_weight': 10, # increase if train-test-gap is large. More conservative but less overfitting
        'max_depth': 6,
        'seed': 42
    }
        
    cv_models, cv_results_train, cv_results_test = cross_validate(features_union_df, params, num_boost_round=13)

    print("********************* TRAIN *********************")
    for metric_lst in cv_metrics(cv_results_train):
        metric_lst = np.array(metric_lst)
        print(f"LB {(metric_lst.mean() - metric_lst.std()):+.4f} to UB {(metric_lst.mean() + metric_lst.std()):+.4f} (± 1σ)")

    print("********************* TEST *********************")
    for metric_lst in cv_metrics(cv_results_test):
        metric_lst = np.array(metric_lst)
        print(f"LB {(metric_lst.mean() - metric_lst.std()):+.4f} to UB {(metric_lst.mean() + metric_lst.std()):+.4f} (± 1σ)")

    print(ENABLED_FEATURES)
    print(params)

    # fig, axes = plt.subplots(4, 2, figsize=(40, 24))
    # for i, (name, booster) in enumerate(cv_models[-1].items()):
    #     ax = axes[i][0]
    #     xgb.plot_importance(booster, ax, importance_type='weight', xlabel=name+' (#times feature appears in a tree)', show_values=False, ylabel=None)
    # for i, (name, booster) in enumerate(cv_models[-1].items()):
    #     ax = axes[i][1]
    #     xgb.plot_importance(booster, ax, importance_type='gain', xlabel=name+' (average gain of splits using this feature)', show_values=False, ylabel=None)

if __name__ == '__main__':
    main()