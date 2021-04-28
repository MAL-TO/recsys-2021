import abc

from metrics import compute_score

class ModelInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'fit') and
                callable(subclass.fit) and
                hasattr(subclass, 'predict') and
                callable(subclass.predict) and
                hasattr(subclass, 'evaluate') and
                callable(subclass.evaluate) and
                hasattr(subclass, 'load_pretrained') and
                callable(subclass.load_pretrained) and
                hasattr(subclass, 'save_to_logs') and
                callable(subclass.save_to_logs) or
                NotImplemented)

    @abc.abstractmethod
    def fit(self, train_data, valid_data, hyperparams):
        """Fit model to given training data and validate it.
        Returns the best model found in validation."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, test_data):
        """Predict test data. Returns predictions."""
        raise NotImplementedError

    def evaluate(self, test_ksdf):
        """Predict test data and evaluate the model on metrics.
        Returns the metrics."""
        predictions_df = self.predict(test_ksdf)

        results = {}
        for column in predictions_df.columns:
            AP, RCE = compute_score(test_ksdf[column].to_numpy(), predictions_df[column].to_numpy())
            results[f'{column}_AP'] = AP
            results[f'{column}_RCE'] = RCE

        return results

    @abc.abstractmethod
    def load_pretrained(self):
        """Load a pretrained model saved in `models/`."""
        raise NotImplementedError

    @abc.abstractmethod
    def save_to_logs(self):
        """Save the results of the latest test performed to logs."""
        raise NotImplementedError
