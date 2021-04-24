import h2o
from h2o.estimators import H2OXGBoostEstimator

# from metrics import...

from model.interface import ModelInterface

class H2OXGBoostBaseline(ModelInterface):
    def __init__(self):
        h2o.init()
        
        is_xgboost_available = H2OXGBoostEstimator.available()
        
        if not is_xgboost_available:
            raise RuntimeError("H2OXGBoostEstimator is not available!")
        
        self.model = None
        self.labels = [
            "reply_timestamp",
            "retweet_timestamp",
            "retweet_with_comment_timestamp",
            "like_timestamp"
        ]
    
    def fit(self, train_data, valid_data, hyperparams):
        """Fit model to given training data and validate it.
        Returns the best model found in validation."""
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
            models[label] = model
        
        # Save (best on valid) trained model
        self.model = models
            
        return models
    
    def predict(self, test_data):
        """Predict test data. Returns predictions."""
        
        test_frame = h2o.H2OFrame(test_data.to_pandas())
        
        predictions = dict()
        for label in self.labels:
            # TODO: as_data_frame() will probably crash on the cluster dataset
            predictions[label] = self.model[label].predict(test_frame).as_data_frame().values.tolist()
            
        return predictions
        
    def evaluate(self, test_data, save_to_logs=False):
        """Predict test data and evaluate the model on metrics.
        Returns the metrics."""
        
        # truth = ???
        preds = self.predict(test_data)
        
        # metrics = compute_metrics(truth, preds) <- this is from module model.metrics.[...]
        metrics = {"metric": None}
        
        self.save_to_logs(metrics)
        
        return metrics
    
    def save_to_logs(self, metrics):
        """Save the results of the latest test performed to logs."""
        pass