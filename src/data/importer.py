import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession


features = [
    # Tweet features
    "text_tokens",      # List[long]    Ordered list of Bert ids corresponding to Bert tokenization of Tweet text
    "hashtags",         # List[string]  Tab separated list of hastags (identifiers) present in the tweet
    "tweet_id",         # String        Tweet identifier (unique)
    "present_media",    # List[String]  Tab separated list of media types. Media type can be in (Photo, Video, Gif)
    "present_links",    # List[string]  Tab separated list of links (identifiers) included in the Tweet
    "present_domains",  # List[string]  Tab separated list of domains included in the Tweet (twitter.com, dogs.com)
    "tweet_type",       # String        Tweet type, can be either Retweet, Quote, Reply, or Toplevel
    "language",         # String        Identifier corresponding to the inferred language of the Tweet
    "tweet_timestamp",  # Long          Unix timestamp, in sec of the creation time of the Tweet

    # Engaged-with User (i.e., Engagee) Features
    "engaged_with_user_id",                 # String    User identifier
    "engaged_with_user_follower_count",     # Long      Number of followers of the user
    "engaged_with_user_following_count",    # Long      Number of accounts the user is following
    "engaged_with_user_is_verified",        # Bool      Is the account verified?
    "engaged_with_user_account_creation",   # Long      Unix timestamp, in seconds, of the creation time of the account

    # Engaging User (i.e., Engager) Features
    "engaging_user_id",                     # String    User identifier
    "engaging_user_follower_count",         # Long      Number of followers of the user
    "engaging_user_following_count",        # Long      Number of accounts the user is following
    "engaging_user_is_verified",            # Bool      Is the account verified?
    "engaging_user_account_creation",       # Long      Unix timestamp, in seconds, of the creation time of the account

    # Engagement features
    "engagee_follows_engager"   # Bool  Does the account of the engaged-with tweet author follow the account that has made the engagement?
]

features_idx = dict(zip(features, range(len(features))))

labels_idx = {
    # Engagement features (cont.)
    "reply_timestamp": 20,                  # Long      Unix timestamp (in seconds) of one of the replies, if there is at least one
    "retweet_timestamp": 21,                # Long      Unix timestamp (in seconds) of the retweet by the engaging user, if there is at least one
    "retweet_with_comment_timestamp": 22,   # Long      Unix timestamp (in seconds) of one of the retweet with comment by the engaging user, if there is at least one
    "like_timestamp": 23                    # Long      Unix timestamp (in seconds) of the like by the engaging user, if they liked the tweet
}

timestamp_cols = [
    "tweet_timestamp",
    "engaged_with_user_account_creation",
    "engaging_user_account_creation",
    "reply_timestamp",
    "retweet_timestamp",
    "retweet_with_comment_timestamp",
    "like_timestamp"
]


def read_csv(path, include_targets):
    names = features
    if include_targets: names += list(labels_idx)
    data = ks.read_csv(path, sep='\x01', names=names)
    return data


def read_parquet(path, include_targets):
    names = features
    if include_targets: names += list(labels_idx)
    data = ks.read_parquet(path, columns=names)
    return data


def import_data(path, include_targets=True):
    extension = path.split('.').pop()

    if extension == 'parquet':
        raw_data = read_parquet(path, include_targets)
    else:
        raw_data = read_csv(path, include_targets)

    return raw_data
