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
        
        self.labels = ["reply", "retweet", "retweet_with_comment", "like"]

        # Specify default and custom features to use in the model
        self.enabled_features = [
            "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaging_user_follower_count",
            "engaging_user_following_count",
        ] + ["TE_user_lang_" + label for label in self.labels]

        # Specify extractors and auxiliaries required by the enabled features
        self.enabled_auxiliaries = []
        self.enabled_extractors = [
            "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaging_user_follower_count",
            "engaging_user_following_count",
            "te_user_lang",
            "binarize_timestamps"
        ]
        if include_targets:
            self.enabled_extractors.append("binarize_timestamps")

    @staticmethod
    def serialized_model_path_for_target(target: str) -> str:
        p = (
            Path(ROOT_DIR)
            / "../serialized_models"
            / f"h2o_xgboost_baseline_{target}.model"
        )
        return str(p.resolve())

    def fit(self, train_data, _valid_data, _hyperparams):
        """Fit model to given training data and validate it.
        Returns the best model found in validation."""

        hc = H2OContext.getOrCreate()
        sdf_train_data = train_data.to_spark()

        train_frame = hc.asH2OFrame(sdf_train_data)

        # TODO: hyperparameter tuning; unbalancement handling?

        models = dict()
        for label in self.labels:
            ignored = set(self.labels) - set(label)
            model = H2OXGBoostEstimator(seed=self.seed)
            model.train(
                y=label,
                ignored_columns=list(ignored),
                training_frame=train_frame
            )
            model.save_mojo(self.serialized_model_path_for_target(label))
            models[label] = model

        # Save (best on valid) trained model
        self.model = models

        return models

    def predict(self, test_data):
        """Predict test data. Returns predictions."""
        schema = StructType(
            [
                StructField("reply", FloatType(), False),
                StructField("retweet", FloatType(), False),
                StructField("retweet_with_comment", FloatType(), False),
                StructField("like", FloatType(), False),
                StructField("tweet_id", StringType(), False),
                StructField("engaging_user_id", StringType(), False),
            ]
        )

        # DataFrame.to_pandas() drops the index, so we need to save it
        # separately and reattach it later.

        # H2OFrame does not provide an index like pandas, but rather appears
        # to have an internal numerical index to preserve ordering.
        # https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/frame.html#h2oframe

        # So we trust that H2O keeps everything in order and drop our custom
        # index ["tweet_id", "engaging_user_id"] in favour of a "standard"
        # numerical index such as 0, 1, 2, ..., only to reattach it later
        # when returning the predictions DataFrame.

        ks_test_data_index = test_data.reset_index(drop=False)
        ks_index = ks_test_data_index[["tweet_id", "engaging_user_id"]]

        h2oframe_test = h2o.H2OFrame(test_data.to_pandas())

        df_predictions = pd.DataFrame()
        for label in self.labels:
            df_predictions[label] = (
                self.model[label].predict(h2oframe_test).as_data_frame()["True"].values
            )

        # Reattach real index (Lord have mercy)
        df_predictions = df_predictions.join(ks_index.to_pandas())
        ks_predictions = ks.DataFrame(df_predictions)

        return ks_predictions.to_spark()

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
