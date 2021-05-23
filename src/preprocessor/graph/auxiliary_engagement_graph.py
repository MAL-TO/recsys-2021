import databricks.koalas as ks
from pyspark.sql.functions import lit

def auxiliary_engagement_graph(raw_data, auxiliary_train=None):
    """
    The engagement graph is a multigraph where:
    - vertices u_i are users
    - edges e_i are engagements

    An edge (u_i, u_j) corresponds to an engagement: user u_i engaged with user
    u_j's tweet in some way. If u_i engaged with user u_j's tweet in multiple
    ways, there will be multiple edges characterized by different engagement
    types.

    Vertices (users) have the following attributes:
    - `id`
    - `follower_count`
    - `following_count`
    - `is_verified`

    Edges (engagements) have the following attributes:
    - `src`
        Engaging user id
    - `dst`
        Engaged with user id
    - `interaction_type`
        Either 1, 2, 3, 4 corresponding respectively to "reply", "retweet",
        "retweet_with_comment", "like"
    - `engagee_follows_engager`
        True if u_j follows u_i, i.e., if the engaged-with user follows the user
        that made the engagement, False otherwise

    TODO(Manuele): check size and eventually aggregate further. Right now, the
        number of rows of sdf_engagements are about half the number of rows of
        raw_data

    TODO(Manuele): also include non-positive samples, as we may be interested
        in predicting that a user does *not* engage a particular tweet

    """

    if auxiliary_train is None:  # Training time: targets are available
        # Reset index is required not to lose `engaging_user_id` and `tweet_id` cols
        sdf_raw_data = raw_data.reset_index(drop=False).to_spark()

        # Filter out negative samples, i.e., non-interaction samples.
        sdf_positives = sdf_raw_data.filter(
            "LENGTH(reply_timestamp) > 0"
            " OR LENGTH(retweet_timestamp) > 0"
            " OR LENGTH(retweet_with_comment_timestamp) > 0"
            " OR LENGTH(like_timestamp) > 0"
        )

        # Extract user (nodes) DataFrame
        sdf_engaged_with_users = sdf_positives.selectExpr(
            "engaged_with_user_id AS id",  # Required column
            "engaged_with_user_follower_count AS follower_count",
            "engaged_with_user_following_count AS following_count",
            "engaged_with_user_is_verified AS is_verified",
            "engaged_with_user_account_creation AS account_creation"
        )

        sdf_engaging_users = sdf_positives.selectExpr(
            "engaging_user_id AS id",  # Required column
            "engaging_user_follower_count AS follower_count",
            "engaging_user_following_count AS following_count",
            "engaging_user_is_verified AS is_verified",
            "engaging_user_account_creation AS account_creation"
        )

        sdf_union_users = sdf_engaged_with_users.union(sdf_engaging_users)
        sdf_users = sdf_union_users.dropDuplicates(['id'])

        # Extract engagements (edges) DataFrame
        engagements_attributes = [
            "engaging_user_id AS src",  # Required column
            "engaged_with_user_id AS dst",  # Required column
            "engagee_follows_engager AS dst_follows_src",
        ]

        # Associates the column of the timestamp of the interaction with the interaction type id
        interaction_types = {
            "reply_timestamp": 1,
            "retweet_timestamp": 2,
            "retweet_with_comment_timestamp": 3,
            "like_timestamp": 4
        }

        sdf_reply_interactions = sdf_positives.selectExpr(
            *engagements_attributes
        ).where("LENGTH(reply_timestamp) > 0").withColumn("interaction_type", lit(interaction_types["reply_timestamp"]))
        sdf_retweet_interactions = sdf_positives.selectExpr(
            *engagements_attributes
        ).where("LENGTH(retweet_timestamp) > 0").withColumn("interaction_type", lit(interaction_types["retweet_timestamp"]))
        sdf_retweet_with_comment_interactions = sdf_positives.selectExpr(
            *engagements_attributes
        ).where("LENGTH(retweet_with_comment_timestamp) > 0").withColumn("interaction_type", lit(interaction_types["retweet_with_comment_timestamp"]))
        sdf_like_interactions = sdf_positives.selectExpr(
            *engagements_attributes
        ).where("LENGTH(like_timestamp) > 0").withColumn("interaction_type", lit(interaction_types["like_timestamp"]))

        sdf_engagements = sdf_reply_interactions.union(sdf_retweet_interactions).union(sdf_retweet_with_comment_interactions).union(sdf_like_interactions)

        # Convert to koalas DataFrame
        ks_users = sdf_users.to_koalas()
        ks_engagements = sdf_engagements.to_koalas()

    else:  # Inference time: targets are unavailable, raw_data is test data
        ks_users_train = auxiliary_train["engagement_graph_users"]
        ks_engagements_train = auxiliary_train["engagement_graph_engagements"]

        # TODO (Manuele): implement extension

        ks_users = ks_users_train
        ks_engagements = ks_engagements_train

    return {
        "engagement_graph_users": ks_users,
        "engagement_graph_engagements": ks_engagements
    }