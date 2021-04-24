import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession

def is_positive(raw_data, features):
    """True if at least one target is not null, i.e., if the engagee
    has engaged with the tweet in some way. False otherwise.
    """
    
    is_positive_col = (
        ks.notna(raw_data["reply_timestamp"]) | \
        ks.notna(raw_data["retweet_timestamp"]) | \
        ks.notna(raw_data["retweet_with_comment_timestamp"]) | \
        ks.notna(raw_data["like_timestamp"])
    )
    
    return is_positive_col
