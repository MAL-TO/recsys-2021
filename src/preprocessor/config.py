import json

class FeatureConfig():
    """Handle feature configuration"""
    
    def __init__(self, config_path):
        """Read configuration file (config.json)

        Args:
            config_path (str): configuration file location
        """
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
    def is_feature_enabled(self, feature_name):
        return {**self.config["default"], **self.config["custom"]}[feature_name]
    
    def get_enabled_features_list(self, type):
        features = [feat for feat in self.config[type] if self.is_feature_enabled(feat)]
        return features
    
    def get_enabled_custom_features_list(self):
        features = self.get_enabled_features_list("custom")
        return features

    def get_enabled_default_features_list(self):
        features = self.get_enabled_features_list("default")
        return features