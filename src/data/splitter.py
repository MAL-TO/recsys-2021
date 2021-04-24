import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession

def naive_split(data):
    """Split data according to its dataframe ordering"""
    
    train_frac = 0.60
    valid_frac = 0.20
    
    len_data = len(data)
    valid_bound = int(len_data * train_frac)
    test_bound = int(len_data * (train_frac + valid_frac))
    
    train_bounds = {'start': 0, 'end': valid_bound}
    valid_bounds = {'start': valid_bound, 'end': test_bound}
    test_bounds = {'start': test_bound, 'end': len_data}
    
    return train_bounds, valid_bounds, test_bounds

def train_valid_test_split_bounds(data):
    return naive_split(data)
