import abc

class ModelInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'fit') and
                callable(subclass.fit) and
                hasattr(subclass, 'predict') and
                callable(subclass.predict) and
                hasattr(subclass, 'evaluate') and
                callable(subclass.evaluate) and
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
        
    @abc.abstractmethod
    def evaluate(self, test_data):
        """Predict test data and evaluate the model on metrics.
        Returns the metrics."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def save_to_logs(self):
        """Save the results of the latest test performed to logs."""
        raise NotImplementedError