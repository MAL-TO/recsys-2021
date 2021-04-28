from model.interface import ModelInterface

import numpy as np
import xgboost as xgb
import pandas as pd

from metrics import compute_score

class Model(ModelInterface):
    def __init__(self):
        self.models = {}
        self.target_columns = [
            "reply", 
            "retweet", 
            "retweet_with_comment",
            "like"
        ]
        self.old_target_columns = [
            "reply_timestamp",
            "retweet_timestamp",
            "retweet_with_comment_timestamp",
            "like_timestamp",
        ]
        default = [
            "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaging_user_follower_count",
            "engaging_user_following_count"
        ]
        custom = [
            "is_positive"
        ]
        
        # For feature store
        self.enabled_features = {"custom": custom, "default":  default + self.old_target_columns}

        # For training and inference
        self.features = custom + default + self.old_target_columns


    def fit(self, train_ksdf, valid_ksdf, _hyperparameters):
        ###############################################################################
        # Convert to koalas dataframes to pandas
        train_df = train_ksdf.to_pandas()
        valid_df = valid_ksdf.to_pandas()
        
        ###############################################################################
        # Convert timestamp labels to binary labels (TODO(Andrea): this should be done
        # on import imho)
        
        for df in [train_df, valid_df]:
            for old_col, new_col in zip(self.old_target_columns, self.target_columns):
                    df[new_col] = df[old_col].notnull()
        
        ###############################################################################
        # Train and save models
        xgb_parameters = {"objective": "binary:logistic", "eval_metric": "logloss"}
        for target in self.target_columns:
            dtrain = xgb.DMatrix(data=train_df[self.features], label=train_df[target])
            model = xgb.train(xgb_parameters, dtrain=dtrain)
            self.models[target] = model

    def predict(self, test_ksdf):
        ###############################################################################
        # Convert to koalas dataframes to pandas
        test_df = test_ksdf.to_pandas()
        
        ###############################################################################
        # Compute predictions for all models
        dtest = xgb.DMatrix(data=test_df[self.features])
        predictions_df = pd.DataFrame()
        for target in self.target_columns:
            predictions_df[target] = self.models[target].predict(dtest)
        
        return predictions_df
    
    def evaluate(self, test_ksdf, save_to_logs=False):
        ###############################################################################
        # Convert to koalas dataframes to pandas
        # TODO(Andrea): since we have fit and predict we do we need evaluate to
        # be implemented in the model?
        predictions_df = self.predict(test_ksdf)
    
        # Convert timestamp labels to binary labels (TODO(Andrea): this should be done
        # on import imho)

        for df in [test_ksdf]:
            for old_col, new_col in zip(self.old_target_columns, self.target_columns):
                    df[new_col] = df[old_col].notnull()

        ###############################################################################
        # Compute metrics for all models 
        results = {}
        for column in predictions_df.columns:
            AP, RCE = compute_score(test_ksdf[column].to_numpy(), predictions_df[column].to_numpy())
            results[f'{column}_AP'] = AP
            results[f'{column}_RCE'] = RCE

        return results

    def save_to_logs(self, metrics):
        pass
