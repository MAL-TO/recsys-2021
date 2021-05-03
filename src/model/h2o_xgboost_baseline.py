import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession

import h2o
from h2o.estimators import H2OXGBoostEstimator

from model.interface import ModelInterface

# Warning: this class only works with *small*-ish datasets, as it casts Spark
# dataframes to Pandas dataframes (H2O natively does not work with Spark).
# Make sure you have enough memory on your machine.

# Tested with the following features enabled:
# "engaged_with_user_follower_count"
# "engaged_with_user_following_count"
# "engaging_user_follower_count"
# "engaging_user_following_count"
# all targets

from pathlib import Path
from constants import ROOT_DIR

class Model(ModelInterface):
    def __init__(self, include_targets=True):
        h2o.init()

        is_xgboost_available = H2OXGBoostEstimator.available()

        if not is_xgboost_available:
            raise RuntimeError("H2OXGBoostEstimator is not available!")

        self.model = None
        self.labels = [
            "reply",
            "retweet",
            "retweet_with_comment",
            "like"
        ]
        self.enabled_features = [
            # Tweet features
            "tweet_type",       # String        Tweet type, can be either Retweet, Quote, Reply, or Toplevel
            "language",         # String        Identifier corresponding to the inferred language of the Tweet
            "tweet_timestamp",  # Long          Unix timestamp, in sec of the creation time of the Tweet

            # Engaged-with User (i.e., Engagee) Features
            "engaged_with_user_follower_count",     # Long      Number of followers of the user
            "engaged_with_user_following_count",    # Long      Number of accounts the user is following
            "engaged_with_user_is_verified",        # Bool      Is the account verified?
            "engaged_with_user_account_creation",   # Long      Unix timestamp, in seconds, of the creation time of the account

            # Engaging User (i.e., Engager) Features
            "engaging_user_follower_count",         # Long      Number of followers of the user
            "engaging_user_following_count",        # Long      Number of accounts the user is following
            "engaging_user_is_verified",            # Bool      Is the account verified?
            "engaging_user_account_creation",       # Long      Unix timestamp, in seconds, of the creation time of the account

            # Engagement features
            "engagee_follows_engager"   # Bool  Does the account of the engaged-with tweet author follow the account that has made the engagement?
        ]
        if include_targets: self.enabled_features += "binarize_timestamps"


    @staticmethod
    def serialized_model_path_for_target(target: str) -> str:
        p = Path(ROOT_DIR) / '../serialized_models' / f'h2o_xgboost_baseline_{target}.model'
        return str(p.resolve())

    def fit(self, train_data, valid_data, hyperparams):
        """Fit model to given training data and validate it.
        Returns the best model found in validation."""

        # Cast to h2o frames
        train_frame = h2o.H2OFrame(train_data.to_pandas())
        valid_frame = h2o.H2OFrame(valid_data.to_pandas())

        # TODO: try to implement a hyperparam tuning inside fit (use
        # additional helper methods in this class if needed, don't
        # modify the interface).
        # Try a grid search and/or a random search.
        # Once a best model has been found, return the best model.

        models = dict()
        for label in self.labels:
            # TODO: handle unbalancement (up/down sampling, other?)
            ignored = set(self.labels) - set(label)
            model = H2OXGBoostEstimator(**hyperparams)
            model.train(y=label,
                        ignored_columns=list(ignored),
                        training_frame=train_frame,
                        validation_frame=valid_frame)
            model.save_mojo(self.serialized_model_path_for_target(label))
            models[label] = model

        # Save (best on valid) trained model
        self.model = models

        return models

    def predict(self, test_data):
        """Predict test data. Returns predictions."""

        test_frame = h2o.H2OFrame(test_data.to_pandas())

        predictions = pd.DataFrame()
        for label in self.labels:
            predictions[label] = self.model[label].predict(test_frame).as_data_frame()['True'].values

        return predictions

    def load_pretrained(self):
        self.model = {}
        for label in self.labels:
            # Select the first model in the directory
            p = str(next(Path(self.serialized_model_path_for_target(label)).iterdir()).resolve())
            self.model[label] = h2o.import_mojo(p)

    def save_to_logs(self, metrics):
        """Save the results of the latest test performed to logs."""
        pass
