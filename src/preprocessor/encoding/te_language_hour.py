import databricks.koalas as ks
# TODO(Francesco): fix suffix mess ==> sometimes result in error
# TODO(Francesco): this is very slow, too many join/merge ...
# Hour of day is not meaningful due to time zones
# What can be useful instead is hour_of_day target encoded with language

# def te_language_hour(raw_data, features = None, auxiliary_dict = None):
#     """
#     Target encoding grouping by language and hour of the day
#     """
#     ALPHA=20 # smoothing factor
#     targets = ["reply", "retweet", "retweet_with_comment", "like"]
#     target_encoded = {}
#     te_df = raw_data.merge(features['hour_of_day'], left_index=True, right_index=True)
#     for target in targets:
#         target_series = features[target].apply(lambda x: 1 if x else 0)
#         te_df = te_df.merge(target_series, left_index=True, right_index=True)
#         # te_df = ks.concat([raw_data, features['hour_of_day'], target_series], axis=1, join="inner")
#         # te_df.set_index(['tweet_id', 'engaging_user_id'], inplace = True)
#         te = ks.sql(f'''SELECT mean({target}) as mean_{target}, count(*) as count_{target}, hour_of_day, language
#         FROM {{te_df}} 
#         GROUP BY hour_of_day, language''')

#         # Apply smoothing factor
#         target_mean = target_series.mean()
#         te[f'te_hour_lang_{target}'] = ((te[f'count_{target}']*te[f'mean_{target}'] + target_mean*ALPHA) / (ALPHA + te[f'count_{target}']))

#         te_df = te_df.reset_index(drop=False).merge(te, how = 'left', on=['hour_of_day', 'language']).set_index(['tweet_id', 'engaging_user_id'])
#         target_encoded[f'te_hour_lang_{target}'] = te_df[f'te_hour_lang_{target}']

#     return target_encoded

import category_encoders as ce


# 200k with te_language_hour
# 
# [Stage 254:==================================================>  (189 + 4) / 200]********************* TRAIN *********************
# LB +0.1032 to UB +0.1343 (± 1σ)
# LB -0.0360 to UB +2.2987 (± 1σ)
# ********************* TEST *********************
# LB +0.0667 to UB +0.0805 (± 1σ)
# LB -4.3184 to UB -1.2539 (± 1σ)
# ['engaged_with_user_follower_count', 'engaged_with_user_following_count', 'engaging_user_follower_count', 'engaging_user_following_count', 'hour_of_day', 'binarize_timestamps', 'te_language_hour']
# {'tree_method': 'hist', 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'subsample': 0.7, 'min_child_weight': 10, 'max_depth': 6, 'seed': 42}
# 