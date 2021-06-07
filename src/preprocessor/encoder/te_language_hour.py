import os
import pickle as pkl
from pysparkling import H2OContext
import databricks.koalas as ks
from h2o.estimators import H2OTargetEncoderEstimator
from constants import PATH_AUXILIARIES

hc = H2OContext.getOrCreate()

def te_language_hour(raw_data, features, auxiliaries, is_inference):

    # auxiliary_path_rw?
    auxiliary_path = os.path.join(PATH_AUXILIARIES, 'te_language_hour')

    # In order to encode on multiple features,
    # make a new categorical feature appending the values of the two features
    categorical_feature = 'language_hour'
    df = ks.concat([raw_data, features['hour_of_day']], axis=1)
    df[categorical_feature] = df.apply(lambda row: str(row.hour_of_day) + "_" + row.language, axis=1)

    targets = ["reply", "retweet", "retweet_with_comment", "like"]
    index_cols = ['tweet_id', 'engaging_user_id']
    target_encoded = {}

    # Can encode more than one col: eventually, we want a single encoding function for all TE
    encoded_columns = [categorical_feature] 

    if is_inference:
        # Define H2O DataFrame
        h2o_frame = hc.asH2OFrame(df.reset_index(drop=False).to_spark())

        # Deserialize encoders
        with open(auxiliary_path, 'rb') as f:
            encoders = pkl.load(f)

        for target in targets:
            new_feature = f'TE_language_hour_{target}'
            te = encoders[target]

            new_col_h2o = te.transform(frame=h2o_frame, as_training=True)[:, index_cols + [categorical_feature + "_te"]]
#             new_col_spark = hc.asDataFrame(new_col_h2o)
#             new_col_koalas = ks.DataFrame(new_col_spark).set_index(index_cols).squeeze()
            new_col_pandas = new_col_h2o.as_data_frame()
            new_col_koalas = ks.from_pandas(new_col_pandas).set_index(index_cols).squeeze()
            new_col_koalas.name = new_feature

        target_encoded[new_feature] = new_col_koalas

    else:
        # Add binary targets to df
        for target in targets:
            df[target] = df[target + "_timestamp"].notna()

        # Define H2O DataFrame
        h2o_frame = hc.asH2OFrame(df.reset_index(drop=False).to_spark())

        encoders = {}
        ALPHA = 20
        NOISE = 0.01 # In general, the less data you have the more regularization you need
        INFLECTION_POINT = 20 # ?

        for target in targets:
            new_feature = f'TE_language_hour_{target}'

            te = H2OTargetEncoderEstimator(
                blending = True,
                noise = NOISE,
                inflection_point = INFLECTION_POINT,
                smoothing=ALPHA
            )

            te.train(x=encoded_columns,
                 y=target,
                 training_frame=h2o_frame)

            # Slicing: indexing columns, new_feature column
            new_col_h2o = te.transform(frame=h2o_frame, as_training=True)[:, index_cols + [categorical_feature + "_te"]]
#             new_col_spark = hc.asDataFrame(new_col_h2o)
#             new_col_koalas = ks.DataFrame(new_col_spark).set_index(index_cols).squeeze()
            new_col_pandas = new_col_h2o.as_data_frame()
            new_col_koalas = ks.from_pandas(new_col_pandas).set_index(index_cols).squeeze()
            new_col_koalas.name = new_feature

            target_encoded[new_feature] = new_col_koalas
            encoders[target] = te

        # Serialize encoders
        with open(auxiliary_path, 'wb+') as f:
            pkl.dump(encoders, f, protocol=pkl.HIGHEST_PROTOCOL)

    return target_encoded


    # if is_inference:
    #     with open(auxiliary_path, 'rb') as f:
    #         encoders = pkl.load(f)

    #     for target in targets:
    #         new_feature = f'TE_language_hour_{target}'
    #         te = encoders[target]
    #         te_result = ks.from_pandas(te.transform(df[categorical_feature].to_pandas())).squeeze()
    #         te_result.name = new_feature
    #         target_encoded[new_feature] = te_result

    # else:
    #     ALPHA=20 # smoothing factor
    #     NOISE = 0.1 # TODO(Francesco) set to 0 for final training and inference
    #     INFLECTION_POINT = 20
    #     encoders = {}

    #     h2o_frame = hc.asH2OFrame(df.to_spark())
    #     for target in targets:
    #         new_feature = f'TE_language_hour_{target}'
    #         # te = ce.TargetEncoder(cols=[categorical_feature], smoothing = ALPHA)
    #         # te_result = ks.from_pandas(te.fit_transform(df[categorical_feature], df[target])).squeeze()
    #         te = H2OTargetEncoderEstimator(blending=True,
    #                                    inflection_point=INFLECTION_POINT,
    #                                    smoothing=ALPHA,
    #                                    noise=NOISE     # In general, the less data you have the more regularization you need
    #                                    )
    #         # Fit the encoder. Can do on multiple cols at tghe time.
    #         # TODO(Francesco): unique encoding function when we have defined the TE features
    #         te.train(
    #             x=[categorical_feature],
    #             y = target,
    #             training_frame=h2o_frame
    #         )
    #         h2o_frame = te.transform(frame=h2o_frame, as_training=True)
    #         te_result.name = new_feature
    #         target_encoded[new_feature] = te_result
    #         encoders[target] = te
    #         print(te_result.describe())

    #     # Store the list of TE for each target
    #     with open(auxiliary_path, 'wb+') as f:
    #         pkl.dump(encoders, f, protocol=pkl.HIGHEST_PROTOCOL)

    # return target_encoded