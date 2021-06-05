import os
import pickle as pkl
import databricks.koalas as ks
import category_encoders as ce
from constants import PATH_AUXILIARIES

def te_tweet_type_word_count(raw_data, features, auxiliaries, is_inference):

    # auxiliary_path_rw?
    auxiliary_path = os.path.join(PATH_AUXILIARIES, 'te_tweet_type_word_count')

    # make new column
    categorical_feature = 'te_tweet_type_word_count'
    df = ks.concat([raw_data, features['word_count']], axis=1)
    df[categorical_feature] = df.apply(lambda row: str(row.word_count) + "_" + row.tweet_type, axis=1)

    targets = ["reply", "retweet", "retweet_with_comment", "like"]
    target_encoded = {}

    if is_inference:
        with open(auxiliary_path, 'rb') as f:
            encoders = pkl.load(f)

        for target in targets:
            new_feature = f'TE_tweet_type_word_count_{target}'
            te = encoders[target]
            te_result = ks.from_pandas(te.transform(df[categorical_feature].to_pandas())).squeeze()
            te_result.name = new_feature
            target_encoded[new_feature] = te_result

    else:
        ALPHA=20 # smoothing factor
        encoders = {}

        df = df.to_pandas()
        for target in targets:
            new_feature = f'TE_tweet_type_word_count_{target}'
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