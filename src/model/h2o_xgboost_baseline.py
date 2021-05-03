import pandas as pd
import os
import contextlib

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
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                h2o.init()
                h2o.no_progress()

        is_xgboost_available = H2OXGBoostEstimator.available()

        if not is_xgboost_available:
            raise RuntimeError("H2OXGBoostEstimator is not available!")

        self.model = None
        self.labels = ["reply", "retweet", "retweet_with_comment", "like"]
        self.enabled_features = [
            # Tweet features
            "tweet_type",
            "language",
            "tweet_timestamp",
            # Engaged-with User (i.e., Engagee) Features
            "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaged_with_user_is_verified",
            "engaged_with_user_account_creation",
            # Engaging User (i.e., Engager) Features
            "engaging_user_follower_count",
            "engaging_user_following_count",
            "engaging_user_is_verified",
            "engaging_user_account_creation",
            # Engagement features
            "engagee_follows_engager",
        ]
        if include_targets:
            self.enabled_features += "binarize_timestamps"

    @staticmethod
    def serialized_model_path_for_target(target: str) -> str:
        p = (
            Path(ROOT_DIR)
            / "../serialized_models"
            / f"h2o_xgboost_baseline_{target}.model"
        )
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
            model.train(
                y=label,
                ignored_columns=list(ignored),
                training_frame=train_frame,
                validation_frame=valid_frame,
            )
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
            predictions[label] = (
                self.model[label]
                .predict(test_frame)
                .as_data_frame()["True"]
                .values
            )

        return predictions

    def load_pretrained(self):
        self.model = {}
        for label in self.labels:
            # Select the first model in the directory
            p = str(
                next(
                    Path(
                        self.serialized_model_path_for_target(label)
                    ).iterdir()
                ).resolve()
            )
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    self.model[label] = h2o.import_mojo(p)

    def save_to_logs(self, metrics):
        """Save the results of the latest test performed to logs."""
        pass
