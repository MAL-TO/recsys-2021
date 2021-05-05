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
    def __init__(self, include_targets=True):
        self.models = {}

        # Default and custom
        self.features = [
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
        self.enabled_features = self.features
        if include_targets:
            self.enabled_features += custom_targets

    @staticmethod
    def serialized_model_path_for_target(target: str) -> str:
        p = (
            Path(ROOT_DIR)
            / "../serialized_models"
            / f"native_xgboost_basline_{target}.model"
        )
        return str(p.resolve())

    def fit(self, train_ksdf, valid_ksdf, _hyperparameters):
        # Convert to koalas dataframes to pandas
        train_df = train_ksdf.to_pandas()
        # Train and save models
        xgb_parameters = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        for target in self.target_columns:
            dtrain = xgb.DMatrix(
                data=train_df[self.features], label=train_df[target]
            )
            model = xgb.train(xgb_parameters, dtrain=dtrain)
            model.save_model(self.serialized_model_path_for_target(target))
            self.models[target] = model

    def predict(self, features_kdf) -> pd.DataFrame:
        with Stage("Converting to DataFrame and caching features_sdf"):
            features_sdf = features_kdf.to_spark(
                index_col=["tweet_id", "engaging_user_id"]
            )
            features_sdf.cache()
            del features_kdf

        with Stage("len(features_sdf)"):
            print("len:", features_sdf.count())

        with Stage("len(features_sdf)"):
            print("len:", features_sdf.count())
            print("#partitions: ", features_sdf.rdd.getNumPartitions())

        with Stage("UDF?"):
            features_with_partition_id_sdf = features_sdf.withColumn(
                "partition_id", spark_partition_id()
            )

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

            # TODO(Andrea): oh god, oh god is this ugly. Fix executor's
            # pythonpath instead.
            @pandas_udf(
                schema,
                PandasUDFType.GROUPED_MAP,
            )
            def predict_udf(features_pdf: pd.DataFrame) -> pd.DataFrame:
                def serialized_model_path_for_target(target: str) -> str:
                    p = (
                        Path(ROOT_DIR)
                        / "../serialized_models"
                        / f"native_xgboost_basline_{target}.model"
                    )
                    return str(p.resolve())

                models = {}
                target_columns = [
                    "reply",
                    "retweet",
                    "retweet_with_comment",
                    "like",
                ]
                features = [
                    "engaged_with_user_follower_count",
                    "engaged_with_user_following_count",
                    "engaging_user_follower_count",
                    "engaging_user_following_count",
                ]
                for target in target_columns:
                    model = xgb.Booster()
                    model.load_model(serialized_model_path_for_target(target))
                    models[target] = model

                # TODO(Andrea): why 200 partitions??
                dfeatures = xgb.DMatrix(data=features_pdf[features])
                for target in target_columns:
                    features_pdf[target] = models[target].predict(dfeatures)

                return features_pdf[
                    ["tweet_id", "engaging_user_id"] + target_columns
                ]

        # TODO(Andrea): should we use withColumn instead?
        return features_with_partition_id_sdf.groupBy("partition_id").apply(
            predict_udf
        )

    def load_pretrained(self):
        for target in self.target_columns:
            model = xgb.Booster()
            model.load_model(self.serialized_model_path_for_target(target))
            self.models[target] = model

    def save_to_logs(self, metrics):
        pass
