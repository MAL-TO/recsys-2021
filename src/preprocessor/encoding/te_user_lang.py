import os
import pickle as pkl
from pysparkling import H2OContext
import databricks.koalas as ks
import h2o
from h2o.estimators import H2OTargetEncoderEstimator
from constants import PATH_AUXILIARIES

hc = H2OContext.getOrCreate()

def te_user_lang(raw_data, features, auxiliaries, is_inference):

    # auxiliary_path_rw?
    auxiliary_path = os.path.join(PATH_AUXILIARIES, 'te_user_lang')
    
    # Binarized targets
    targets = ["reply", "retweet", "retweet_with_comment", "like"]
    index_cols = ['tweet_id', 'engaging_user_id']
    target_encoded = {}
    
    # make new column. We need to access index values
    categorical_feature = 'te_user_lang'
#     cols = ["language"] + [target + "_timestamp" for target in targets]
    df = raw_data
    df[categorical_feature] = df.reset_index(drop=False).\
            apply(lambda row: [row.tweet_id, row.engaging_user_id, row.engaging_user_id + "_" + row.language], axis=1, result_type ='expand').\
            rename({0:'tweet_id', 1:'engaging_user_id', 2: categorical_feature}, axis=1).\
            set_index(['tweet_id', 'engaging_user_id']).squeeze()

    # Can encode more than one col: eventually, we want a single encoding function for all TE
    encoded_columns = [categorical_feature] 
    
    if is_inference:
        # Define H2O DataFrame
        print(df[categorical_feature].head())
        h2o_frame = hc.asH2OFrame(df.reset_index(drop=False).to_spark())
        h2o_frame[categorical_feature] = h2o_frame[categorical_feature].asfactor()
        
        for target, f in zip(targets, os.listdir(auxiliary_path)):
            new_feature = f'TE_user_lang_{target}'
            
            # Deserialize encoders
            te = h2o.load_model(os.path.join(auxiliary_path, f))

            print(te.transform(frame=h2o_frame).head())
            print(te.transform(frame=h2o_frame).columns)
            new_col_h2o = te.transform(frame=h2o_frame)[:, index_cols + [categorical_feature + "_te"]]
#             new_col_spark = hc.asDataFrame(new_col_h2o)
#             new_col_koalas = ks.DataFrame(new_col_spark).set_index(index_cols).squeeze()
            new_col_pandas = new_col_h2o.as_data_frame()
            new_col_koalas = ks.from_pandas(new_col_pandas).set_index(index_cols).squeeze()
            new_col_koalas.name = new_feature
            
            print(new_col_koalas.head())

        target_encoded[new_feature] = new_col_koalas

    else:
        # Add binary targets to df
        for target in targets:
            df[target] = df[target + "_timestamp"].notna()

        # Define H2O DataFrame
        h2o_frame = hc.asH2OFrame(df.reset_index(drop=False).to_spark())
        h2o_frame[categorical_feature] = h2o_frame[categorical_feature].asfactor()

        ALPHA = 20
        NOISE = 0.01 # In general, the less data you have the more regularization you need
        INFLECTION_POINT = 20 # ?

        for target in targets:
            new_feature = f'TE_user_lang_{target}'

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



# ********************* TRAIN *********************                               
# LB +0.2965 to UB +0.3136 (± 1σ)
# LB +12.9521 to UB +14.9195 (± 1σ)
# ********************* TEST *********************
# LB +0.2472 to UB +0.2661 (± 1σ)
# LB +9.4952 to UB +11.8348 (± 1σ)
# ['engaged_with_user_follower_count', 'engaged_with_user_following_count', 'engaging_user_follower_count', 'engaging_user_following_count', 'engaging_user_lang', 'binarize_timestamps']
# {'tree_method': 'hist', 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'subsample': 0.7, 'min_child_weight': 10, 'max_depth': 6, 'seed': 42}