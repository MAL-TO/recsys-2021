import databricks.koalas as ks

# Problema: issue with cross validation ==> I am passing the already processed data, which is not good.
# In fact, validation data in TE see the target, this is why the boost so much

import category_encoders as ce

def tweet_type_word_count(raw_data, features = None, auxiliary_dict = None):

    # make new column
    categorical_feature = 'tweet_type_word_count'
    df = ks.concat([raw_data, features['word_count']], axis=1)
    df[categorical_feature] = df.apply(lambda row: str(row.word_count) + "_" + row.tweet_type, axis=1)

    ALPHA=20 # smoothing factor
    targets = ["reply", "retweet", "retweet_with_comment", "like"]
    target_encoded = {}

    df = df.to_pandas()
    for target in targets:
        new_feature = f'TE_tweet_type_word_count_{target}'
        df[target] = df[target + "_timestamp"].notna() # faster than joining features[target]
        te = ce.TargetEncoder(cols=[categorical_feature], smoothing = ALPHA)
        te_result = ks.from_pandas(te.fit_transform(df[categorical_feature], df[target])).squeeze()
        te_result.name = new_feature
        target_encoded[new_feature] = te_result
        print(te_result.describe())

    return target_encoded