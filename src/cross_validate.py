# TODO(Francesco): DO NOT COMMIT; Andrea's made its own one, I am just copying it on this branch
import h2o
import xgboost as xgb
import databricks.koalas as ks
from h2o.estimators import H2OXGBoostEstimator
from sklearn.model_selection import TimeSeriesSplit

from metrics import compute_score
from preprocessor.features_store import FeatureStore
from util import rm_dir_contents, Stage
from constants import (
    PATH_PREPROCESSED,
    PATH_PREPROCESSED_CLUSTER,
    PATH_AUXILIARIES,
    PATH_AUXILIARIES_CLUSTER,
    MODEL_SEED
)

"""
Split the dataset into time series
I end up with N splits:
1. Train on split 1, validate on split 2
2. train on splits 1 and 2, and validate on split 2, ...
3. train on splits up to N-1, and validate on split N

4. Average performance and use std as a measure of uncertainty.
"""

"""
Dataset: 
Process data with a FeatureStore instance
Once I have my dataset processed, I need to pass it to cross_validate
Then, I will have my metrics returned, which I need to visualize
Please, keep a baseline of the results without additional features for immediate comparison

TODO(Francesco): set up the logging of features, model, scores
"""

def cross_validate(
    raw_data,
    params,
    enabled_features,
    enabled_auxiliaries,
    num_boost_round
):
    df  = raw_data.to_pandas()
    tscv = TimeSeriesSplit(n_splits=3)

    targets = [
            "reply",
            "retweet",
            "retweet_with_comment",
            "like",
    ]

    cv_models = []
    cv_results_train = []
    cv_results_test = []
    for train_ixs, test_ixs in tscv.split(df):
        train_raw = df.iloc[train_ixs]
        test_raw = df.iloc[test_ixs]

        with Stage("Cleaning feature store..."):
            rm_dir_contents(PATH_PREPROCESSED)
            rm_dir_contents(PATH_AUXILIARIES)

        with Stage("Assembling train dataset..."):
            store = FeatureStore(
                PATH_PREPROCESSED,
                PATH_PREPROCESSED_CLUSTER,
                enabled_features,
                PATH_AUXILIARIES,
                PATH_AUXILIARIES_CLUSTER,
                enabled_auxiliaries,
                ks.from_pandas(train_raw),
                is_cluster=False,
                is_inference=False,
            )
            train_split_df = store.get_dataset().to_pandas()

        with Stage("Cleaning feature store..."):
            rm_dir_contents(PATH_PREPROCESSED)

        with Stage("Assembling test dataset..."):
            store = FeatureStore(
                PATH_PREPROCESSED,
                PATH_PREPROCESSED_CLUSTER,
                enabled_features,
                PATH_AUXILIARIES,
                PATH_AUXILIARIES_CLUSTER,
                enabled_auxiliaries,
                ks.from_pandas(test_raw),
                is_cluster=False,
                is_inference=True,
            )
            test_split_df = store.get_dataset().to_pandas()

        features = list(set(train_split_df.columns) - set(targets))
        
        X_train = train_split_df.loc[:, features]
        X_test = test_split_df.loc[:, features]

        # train model
        dtest = xgb.DMatrix(X_test)
        results_train = {}
        results_test = {}
        models = {}
        for target in targets:
            dtrain = xgb.DMatrix(X_train, train_split_df[target])
            models[target] = model = xgb.train(
                params=params,
                num_boost_round=num_boost_round,
                dtrain=dtrain,
                verbose_eval=False
            )
            AP, RCE = compute_score(train_split_df[target], model.predict(dtrain))
            results_train[f"{target}_AP"] = AP
            results_train[f"{target}_RCE"] = RCE

            # Validate the model
            AP, RCE = compute_score(test_split_df[target], model.predict(dtest))
            results_test[f"{target}_AP"] = AP
            results_test[f"{target}_RCE"] = RCE

        cv_models.append(models)
        cv_results_train.append(results_train)
        cv_results_test.append(results_test)

    return cv_models, cv_results_train, cv_results_test


def cv_metrics(cv_results):
    mAPs = []
    mRCEs = []
    for result in cv_results:
        mAP = (
            result['retweet_AP']
            +result['reply_AP']
            +result['like_AP']
            +result['retweet_with_comment_AP']
        ) / 4
        mAPs.append(mAP)
        mRCE = (
            result['retweet_RCE']
            +result['reply_RCE']
            +result['like_RCE']
            +result['retweet_with_comment_RCE']
        ) / 4
        mRCEs.append(mRCE)
    
    return mAPs, mRCEs