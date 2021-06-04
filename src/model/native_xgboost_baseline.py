from pyspark.sql.functions import pandas_udf, spark_partition_id, PandasUDFType
from pyspark.sql.types import (
    FloatType,
    StringType,
    StructType,
    StructField,
)
import xgboost as xgb
import pandas as pd

from model.interface import ModelInterface
from pathlib import Path
from constants import ROOT_DIR
from util import Stage


class Model(ModelInterface):
    def __init__(self, include_targets=True, seed=None):
        self.models = {}
        self.seed = seed

        # Specify default and custom features to use in the model
        # Note: you should copy this list manually into self.predict
        self.features = [
            "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaging_user_follower_count",
            "engaging_user_following_count",
        ]

        # Must be coherent with columns in custom_targets!
        self.target_columns = [
            "reply",
            "retweet",
            "retweet_with_comment",
            "like",
        ]

        # Custom features used as target
        custom_targets = ["binarize_timestamps"]

        # For feature store
        self.enabled_features = self.features + (
            custom_targets if include_targets else []
        )

        # Specify extractors and auxiliaries required by the enabled features
        self.enabled_auxiliaries = []

        self.enabled_extractors = [
            "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaging_user_follower_count",
            "engaging_user_following_count",
            "binarize_timestamps",
        ]

    @staticmethod
    def serialized_model_path_for_target(target: str) -> str:
        p = (
            Path(ROOT_DIR)
            / "../serialized_models"
            / f"native_xgboost_baseline_{target}.model"
        )
        return str(p.resolve())

    def fit(self, train_kdf, _valid_kdf, _hyperparameters):
        train_pdf = train_kdf.to_pandas()

        xgb_parameters = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        for target in self.target_columns:
            dtrain = xgb.DMatrix(data=train_pdf[self.features], label=train_pdf[target])
            model = xgb.train(xgb_parameters, dtrain=dtrain)
            model.save_model(self.serialized_model_path_for_target(target))
            self.models[target] = model

    def predict(self, features_kdf):
        # Schema for the UDF
        schema = StructType(
            [
                StructField("tweet_id", StringType(), False),
                StructField("engaging_user_id", StringType(), False),
                StructField("reply", FloatType(), False),
                StructField("retweet", FloatType(), False),
                StructField("retweet_with_comment", FloatType(), False),
                StructField("like", FloatType(), False),
            ]
        )

        # Variables that need to be local so that we don't use `self.` in the UDF
        # There are pickling + classpath issues with using `self.` inside the UDF since
        # it needs to be serialized/deserialized when sent to the executors
        models = {}
        for target in self.target_columns:
            model = xgb.Booster()
            model.load_model(self.serialized_model_path_for_target(target))
            models[target] = model
        target_columns = self.target_columns
        real_features = [
            "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaging_user_follower_count",
            "engaging_user_following_count",
        ]

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def predict_udf(features_pdf: pd.DataFrame) -> pd.DataFrame:
            dfeatures = xgb.DMatrix(data=features_pdf[real_features])
            for target in target_columns:
                features_pdf[target] = models[target].predict(dfeatures)

            return features_pdf[["tweet_id", "engaging_user_id"] + target_columns]

        with Stage("Converting to DataFrame and caching features_sdf"):
            features_sdf = features_kdf.to_spark(
                index_col=["tweet_id", "engaging_user_id"]
            )
            features_sdf.cache()
            del features_kdf

        with Stage("Compute predictions"):
            features_with_partition_id_sdf = features_sdf.withColumn(
                "partition_id", spark_partition_id()
            )

        # TODO(Andrea): why 200 partitions??
        return features_with_partition_id_sdf.groupBy("partition_id").apply(predict_udf)

    def load_pretrained(self):
        for target in self.target_columns:
            model = xgb.Booster()
            model.load_model(self.serialized_model_path_for_target(target))
            self.models[target] = model

    def save_to_logs(self, metrics):
        pass
