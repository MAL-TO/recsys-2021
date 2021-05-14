import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from metrics import compute_score
from import_data import targets


def cross_validate(train_df, transformer, params, num_boost_round):
    tscv = TimeSeriesSplit(n_splits=5)

    cv_models = []
    cv_results_train = []
    cv_results_test = []

    for train_ixs, test_ixs in tscv.split(train_df):
        train_split_df = train_df.iloc[train_ixs]
        test_split_df = train_df.iloc[test_ixs]

        train_features_df = transformer.fit_transform(train_split_df)
        test_features_df = transformer.transform(test_split_df.drop(targets, axis=1))

        # train model
        dtest = xgb.DMatrix(test_features_df)
        results_train = {}
        results_test = {}
        models = {}
        for target in targets:
            dtrain = xgb.DMatrix(train_features_df, train_split_df[target])
            models[target] = model = xgb.train(
                params=params,
                num_boost_round=num_boost_round,
                dtrain=dtrain,
                verbose_eval=False
            )
            AP, RCE = compute_score(train_split_df[target], model.predict(dtrain))
            results_train[f"{target}_AP"] = AP
            results_train[f"{target}_RCE"] = RCE

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