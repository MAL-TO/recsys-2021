from pyspark.sql.types import (
    FloatType,
    StringType,
    StructType,
    StructField,
)

import pandas as pd
import databricks.koalas as ks
import os
import contextlib

import h2o
from h2o.estimators import H2OXGBoostEstimator
from pysparkling import H2OContext

from metrics import compute_score
from model.interface import ModelInterface

from pathlib import Path
from constants import ROOT_DIR


class Model(ModelInterface):
    def __init__(self, include_targets=True, seed=None):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                h2o.init()
                h2o.no_progress()

        is_xgboost_available = H2OXGBoostEstimator.available()

        if not is_xgboost_available:
            raise RuntimeError("H2OXGBoostEstimator is not available!")

        self.model = None
        self.seed = seed

        # Specify default and custom features to use in the model
        self.enabled_features = [
            "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaging_user_follower_count",
            "engaging_user_following_count",
        ]

        self.labels = ["reply", "retweet", "retweet_with_comment", "like"]

        # Specify extractors and auxiliaries required by the enabled features
        self.enabled_auxiliaries = []
        self.enabled_extractors = [
            "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaging_user_follower_count",
            "engaging_user_following_count",
            "binarize_timestamps",
        ]
        if include_targets:
            self.enabled_extractors.append("binarize_timestamps")

    @staticmethod
    def serialized_model_path_for_target(target: str) -> str:
        p = (
            Path(ROOT_DIR)
            / "../serialized_models"
            / f"h2o_xgboost_pysparkling_{target}.model"
        )
        return str(p.resolve())

    def fit(self, train_data, _valid_data, _hyperparams):
        """Fit model to given training data and validate it.
        Returns the best model found in validation."""

        hc = H2OContext.getOrCreate()

        # to_spark() drops the index
        train_frame = hc.asH2OFrame(train_data.to_spark())

        # TODO: hyperparameter tuning; unbalancement handling?

        models = dict()
        for label in self.labels:
            model = H2OXGBoostEstimator(seed=self.seed)

            ignored = set(self.labels) - set(label)

            model.train(
                y=label, ignored_columns=list(ignored), training_frame=train_frame
            )

            model.save_mojo(self.serialized_model_path_for_target(label))
            models[label] = model

        # Save (best on valid) trained model
        self.model = models

        return models

    def predict(self, test_data):
        """Predict test data. Returns predictions."""
        hc = H2OContext.getOrCreate()

        index_col = ["tweet_id", "engaging_user_id"]

        # H2O guarantees same row ordering
        # https://github.com/h2oai/sparkling-water/issues/194
        sdf_test_data = test_data.to_spark(index_col=index_col)
        test_frame = hc.asH2OFrame(sdf_test_data)
        test_predictions = test_frame[:, ["tweet_id", "engaging_user_id"]]

        for label in self.labels:
            test_predictions = test_predictions.cbind(
                self.model[label].predict(test_frame).rename(columns={"predict": label})
            )

        return hc.asSparkFrame(test_predictions)

    def evaluate(self, test_kdf):
        """Predict test data and evaluate the model on metrics.
        Returns the metrics."""
        target_columns = [
            "reply",
            "retweet",
            "retweet_with_comment",
            "like",
        ]

        predictions_sdf = self.predict(test_kdf)
        predictions_kdf = (
            ks.DataFrame(predictions_sdf)
            .rename(columns={col: ("predicted_" + col) for col in target_columns})
            .set_index(keys=["tweet_id", "engaging_user_id"])
        )

        joined_kdf = predictions_kdf.join(
            right=test_kdf[target_columns].astype("int32"), how="inner"
        )

        results = {}
        for column in target_columns:
            AP, RCE = compute_score(
                joined_kdf[column].to_numpy(),
                joined_kdf["predicted_" + column].to_numpy(),
            )
            results[f"{column}_AP"] = AP
            results[f"{column}_RCE"] = RCE

        return results

    def load_pretrained(self):
        self.model = {}
        for label in self.labels:
            # Select the first model in the directory
            p = str(
                next(
                    Path(self.serialized_model_path_for_target(label)).iterdir()
                ).resolve()
            )
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    self.model[label] = h2o.import_mojo(p)

    def save_to_logs(self, metrics):
        """Save the results of the latest test performed to logs."""
        pass
