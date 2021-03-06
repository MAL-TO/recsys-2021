import databricks.koalas as ks

features = [
    # Tweet features
    # List[long]: Ordered list of Bert ids corresponding to Bert tokenization
    # of Tweet text
    "text_tokens",
    # List[string]: Tab separated list of hastags (identifiers) present in the
    # tweet
    "hashtags",
    # String: Tweet identifier (unique)
    "tweet_id",
    # List[String]: Tab separated list of media types. Media type can be in
    # (Photo, Video, Gif)
    "present_media",
    # List[string]: Tab separated list of links (identifiers) included in the
    # Tweet
    "present_links",
    # List[string]: Tab separated list of domains included in the Tweet
    # twitter.com, dogs.com)
    "present_domains",
    # String: Tweet type, can be either Retweet, Quote, Reply, or Toplevel
    "tweet_type",
    # String: Identifier corresponding to the inferred language of the Tweet
    "language",
    # Long: Unix timestamp, in sec of the creation time of the Tweet
    "tweet_timestamp",
    #
    # Engaged-with User (i.e., Engagee) Features
    # String: User identifier
    "engaged_with_user_id",
    # Long: Number of followers of the user
    "engaged_with_user_follower_count",
    # Long: Number of accounts the user is following
    "engaged_with_user_following_count",
    # Bool: Is the account verified?
    "engaged_with_user_is_verified",
    # Long: Unix timestamp, in seconds, of the creation time of the account
    "engaged_with_user_account_creation",
    # Engaging User (i.e., Engager) Features
    #
    # String: User identifier
    "engaging_user_id",
    # Long: Number of followers of the user
    "engaging_user_follower_count",
    # Long: Number of accounts the user is following
    "engaging_user_following_count",
    # Bool: Is the account verified?
    "engaging_user_is_verified",
    # Long: Unix timestamp, in seconds, of the creation time of the account
    "engaging_user_account_creation",
    #
    # Engagement features
    # Bool: Does the account of the engaged-with tweet author follow the
    # account
    # that has made the engagement?
    "engagee_follows_engager",
]

features_idx = dict(zip(features, range(len(features))))

labels_idx = {
    # Long: Unix timestamp (in seconds) of one of the replies, if there is at
    # least one
    "reply_timestamp": 20,
    # Long: Unix timestamp (in seconds) of the retweet by the engaging user,
    # if there is at least one
    "retweet_timestamp": 21,
    # Long: Unix timestamp (in seconds) of one of the retweet with comment by
    # the engaging user, if there is at least one
    "retweet_with_comment_timestamp": 22,
    # Long: Unix timestamp (in seconds) of the like by the engaging user,
    # if they liked the tweet
    "like_timestamp": 23,
}

timestamp_cols = [
    "tweet_timestamp",
    "engaged_with_user_account_creation",
    "engaging_user_account_creation",
    "reply_timestamp",
    "retweet_timestamp",
    "retweet_with_comment_timestamp",
    "like_timestamp",
]


def read_csv(path, include_targets):
    targets = []
    if include_targets:
        targets = list(labels_idx)
    data = ks.read_csv(
        path,
        sep="\x01",
        names=features+targets,
        index_col=["tweet_id", "engaging_user_id"],
    )
    return data


def read_parquet(path, include_targets):
    targets = []
    if include_targets:
        targets = list(labels_idx)
    data = ks.read_parquet(
        path, columns=features+targets, index_col=["tweet_id", "engaging_user_id"]
    )
    return data


def import_data(path, include_targets=True):
    extension = path.split(".").pop()

    if extension == "parquet":
        raw_data = read_parquet(path, include_targets)
    else:
        raw_data = read_csv(path, include_targets)

    return raw_data
