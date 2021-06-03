import databricks.koalas as ks

# Hour of day is not meaningful due to time zones
# What can be useful instead is hour_of_day target encoded with language

# NOTE(Francesco): cast to pandas and back to koalas is slow but needed..
# Very slow, 2mins for 200k ==> 1h for 10M

import category_encoders as ce

def te_language_hour(raw_data, features = None, auxiliary_dict = None):

    # make new column
    categorical_feature = 'hour_lang'
    df = ks.concat([raw_data, features['hour_of_day']], axis=1)
    df[categorical_feature] = df.apply(lambda row: str(row.hour_of_day) + "_" + row.language, axis=1)

    ALPHA=20 # smoothing factor
    targets = ["reply", "retweet", "retweet_with_comment", "like"]
    target_encoded = {}

    df = df.to_pandas()
    for target in targets:
        new_feature = f'TE_hour_lang_{target}'
        df[target] = df[target + "_timestamp"].notna() # faster than joining features[target]
        te = ce.TargetEncoder(cols=[categorical_feature], smoothing = ALPHA)
        te_result = ks.from_pandas(te.fit_transform(df[categorical_feature], df[target])).squeeze()
        te_result.name = new_feature
        target_encoded[new_feature] = te_result
        print(te_result.describe())

    return target_encoded

# 200k with te_language_hour
# 
# ********************* TRAIN *********************                               
# LB +0.1033 to UB +0.1361 (± 1σ)
# LB -0.0068 to UB +2.3364 (± 1σ)
# ********************* TEST *********************
# LB +0.0670 to UB +0.0792 (± 1σ)
# LB -4.1530 to UB -1.3679 (± 1σ)
# ['engaged_with_user_follower_count', 'engaged_with_user_following_count', 'engaging_user_follower_count', 'engaging_user_following_count', 'hour_of_day', 'binarize_timestamps', 'te_language_hour']
# {'tree_method': 'hist', 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'subsample': 0.7, 'min_child_weight': 10, 'max_depth': 6, 'seed': 42}
# 