import os
import pickle as pkl
from pysparkling import H2OContext
import databricks.koalas as ks
import h2o
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

        for target, f in zip(targets, os.listdir(auxiliary_path)):
            new_feature = f'TE_language_hour_{target}'
            
            # Deserialize encoders
            te = h2o.load_model(os.path.join(auxiliary_path, f))

            new_col_h2o = te.transform(frame=h2o_frame)[:, index_cols + [categorical_feature + "_te"]]
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

            te.download_model(auxiliary_path)

    return target_encoded
