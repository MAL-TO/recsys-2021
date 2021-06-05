import databricks.koalas as ks
import category_encoders as ce

# NOTE(Francesco): cast to pandas and back to koalas is slow but needed..
# Very slow, 2mins for 200k ==> 1h for 10M
def engaging_user_lang(raw_data, features = None, auxiliary_dict = None):

    # make new column. Take care of index: we remove it and restore it back
    categorical_feature = 'user_id_lang'
    raw_data[categorical_feature] = raw_data.reset_index(drop=False).\
            apply(lambda row: [row.tweet_id, row.engaging_user_id, str(row.engaging_user_id) + "_" + row.language], axis=1, result_type ='expand').\
            rename({0:'tweet_id', 1:'engaging_user_id', 2: categorical_feature}, axis=1).\
            set_index(['tweet_id', 'engaging_user_id']).squeeze()

    ALPHA=20 # smoothing factor
    targets = ["reply", "retweet", "retweet_with_comment", "like"]
    target_encoded = {}

    df = raw_data.to_pandas()
    for target in targets:
        new_feature = f'TE_user_lang_{target}'
        df[target] = df[target + "_timestamp"].notna() # faster than joining features[target]
        te = ce.TargetEncoder(cols=[categorical_feature], smoothing = ALPHA)
        te_result = ks.from_pandas(te.fit_transform(df[categorical_feature], df[target])).squeeze()
        te_result.name = new_feature
        target_encoded[new_feature] = te_result
        print(te_result.describe())

    return target_encoded

# ********************* TRAIN *********************                               
# LB +0.2965 to UB +0.3136 (± 1σ)
# LB +12.9521 to UB +14.9195 (± 1σ)
# ********************* TEST *********************
# LB +0.2472 to UB +0.2661 (± 1σ)
# LB +9.4952 to UB +11.8348 (± 1σ)
# ['engaged_with_user_follower_count', 'engaged_with_user_following_count', 'engaging_user_follower_count', 'engaging_user_following_count', 'engaging_user_lang', 'binarize_timestamps']
# {'tree_method': 'hist', 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'subsample': 0.7, 'min_child_weight': 10, 'max_depth': 6, 'seed': 42}