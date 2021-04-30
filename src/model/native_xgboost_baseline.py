from model.interface import ModelInterface

import numpy as np
import xgboost as xgb
import pandas as pd

from pathlib import Path
from constants import ROOT_DIR

class Model(ModelInterface):
    def __init__(self):
        self.models = {}

        # Default and custom
        self.features =[
            "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaging_user_follower_count",
            "engaging_user_following_count"
        ]

        # Must be coherent with columns in custom_targets!
        self.target_columns = ["reply", "retweet", "retweet_with_comment", "like"]

        # Custom features used as target
        custom_targets = [
            "binarize_timestamps"
        ]

        # For feature store
        self.enabled_features = self.features + custom_targets


    @staticmethod
    def serialized_model_path_for_target(target: str) -> str:
        p = Path(ROOT_DIR) / '../serialized_models' / f'native_xgboost_basline_{target}.model'
        return p.resolve()

    def fit(self, train_ksdf, valid_ksdf, _hyperparameters):
        ###############################################################################
        # Convert to koalas dataframes to pandas
        train_df = train_ksdf.to_pandas()
        valid_df = valid_ksdf.to_pandas()
        ###############################################################################
        # Train and save models
        xgb_parameters = {"objective": "binary:logistic", "eval_metric": "logloss"}
        for target in self.target_columns:
            dtrain = xgb.DMatrix(data=train_df[self.features], label=train_df[target])
            model = xgb.train(xgb_parameters, dtrain=dtrain)
            model.save_model(self.serialized_model_path_for_target(target))
            self.models[target] = model


    def predict(self, test_ksdf) -> pd.DataFrame:
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

    def load_pretrained(self):
        for target in self.target_columns:
            model = xgb.Booster()
            model.load_model(self.serialized_model_path_for_target(target))
            self.models[target] = model

    def save_to_logs(self, metrics):
        pass
