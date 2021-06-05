import os
import pickle as pkl
import databricks.koalas as ks
import category_encoders as ce
from constants import PATH_AUXILIARIES

def te_language_hour(raw_data, features, auxiliaries, is_inference):

    # auxiliary_path_rw?
    auxiliary_path = os.path.join(PATH_AUXILIARIES, 'te_language_hour')

    # make new column
    categorical_feature = 'te_language_hour'
    df = ks.concat([raw_data, features['hour_of_day']], axis=1)
    df[categorical_feature] = df.apply(lambda row: str(row.hour_of_day) + "_" + row.language, axis=1)

    targets = ["reply", "retweet", "retweet_with_comment", "like"]
    target_encoded = {}

    if is_inference:
        with open(auxiliary_path, 'rb') as f:
            encoders = pkl.load(f)

        for target in targets:
            new_feature = f'TE_language_hour_{target}'
            te = encoders[target]
            te_result = ks.from_pandas(te.transform(df[categorical_feature].to_pandas())).squeeze()
            te_result.name = new_feature
            target_encoded[new_feature] = te_result

    else:
        ALPHA=20 # smoothing factor
        encoders = {}

        df = df.to_pandas()
        for target in targets:
            new_feature = f'TE_language_hour_{target}'
            df[target] = df[target + "_timestamp"].notna() # faster than joining features[target]
            te = ce.TargetEncoder(cols=[categorical_feature], smoothing = ALPHA)
            te_result = ks.from_pandas(te.fit_transform(df[categorical_feature], df[target])).squeeze()
            te_result.name = new_feature
            target_encoded[new_feature] = te_result
            encoders[target] = te
            print(te_result.describe())

        # Store the list of TE for each target
        with open(auxiliary_path, 'wb+') as f:
            pkl.dump(encoders, f, protocol=pkl.HIGHEST_PROTOCOL)

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