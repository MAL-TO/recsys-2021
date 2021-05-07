import abc

from metrics import compute_score


class ModelInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "fit")
            and callable(subclass.fit)
            and hasattr(subclass, "predict")
            and callable(subclass.predict)
            and hasattr(subclass, "evaluate")
            and callable(subclass.evaluate)
            and hasattr(subclass, "load_pretrained")
            and callable(subclass.load_pretrained)
            and hasattr(subclass, "save_to_logs")
            and callable(subclass.save_to_logs)
            or NotImplemented
        )

    @abc.abstractmethod
    def fit(self, train_data, valid_data, hyperparams):
        """Fit model to given training data and validate it.
        Returns the best model found in validation."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, test_data):
        """Predict test data. Returns predictions."""
        raise NotImplementedError

    def evaluate(self, test_kdf):
        """Predict test data and evaluate the model on metrics.
        Returns the metrics."""
        target_columns = [
            "reply",
            "retweet",
            "retweet_with_comment",
            "like",
        ]

        predictions_kdf = (
            self.predict(test_kdf)
            .to_koalas(index_col=["tweet_id", "engaging_user_id"])
            .rename(columns={col: ("predicted_" + col) for col in target_columns})
        )
        joined_kdf = predictions_kdf.join(right=test_kdf[target_columns], how="inner")

        results = {}
        for column in target_columns:
            AP, RCE = compute_score(
                joined_kdf[column].to_numpy(),
                joined_kdf["predicted_" + column].to_numpy(),
            )
            results[f"{column}_AP"] = AP
            results[f"{column}_RCE"] = RCE

        return results

    @abc.abstractmethod
    def load_pretrained(self):
        """Load a pretrained model saved in `models/`."""
        raise NotImplementedError

    @abc.abstractmethod
    def save_to_logs(self):
        """Save the results of the latest test performed to logs."""
        raise NotImplementedError
