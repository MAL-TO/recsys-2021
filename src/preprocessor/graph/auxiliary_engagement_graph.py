import databricks.koalas as ks
from pyspark.sql.functions import lit


def auxiliary_engagement_graph(raw_data, auxiliary_train=None):
    """
    The engagement graph is composed by a series of multigraphs where:
    - vertices u_i are users
    - edges e_i are engagements

    An edge (u_i, u_j) corresponds to an engagement: user u_i engaged with (or
    did not engage with) user u_j's tweet in some way.

    There exists one multigraph for positive interactions, and one multigraph
    for negative samples (edge src->dst exists if src did not engage with dst
    in any way).

    Vertices (users) have the following attributes:
    - `id`

    Edges (engagements) have the following attributes:
    - `src`
        Engaging user id
    - `dst`
        Engaged with user id

    """

    if auxiliary_train is None:  # Training time: targets are available
        index_col = ["tweet_id", "engaging_user_id"]
        sdf_raw_data = raw_data.to_spark(index_col=index_col)

        # Extract user (nodes) DataFrame
        sdf_engaged_with_users = sdf_raw_data.selectExpr(
            "engaged_with_user_id AS id",  # Required column
        )

        sdf_engaging_users = sdf_raw_data.selectExpr(
            "engaging_user_id AS id",  # Required column
        )

        sdf_union_users = sdf_engaged_with_users.union(sdf_engaging_users)
        sdf_users = sdf_union_users.dropDuplicates(["id"])

        # Extract engagements (edges) DataFrame
        engagements_attributes = [
            "engaging_user_id AS src",  # Required column
            "engaged_with_user_id AS dst",  # Required column
        ]

        # Positive samples (interactions)
        sdf_interactions = (
            sdf_raw_data.selectExpr(*engagements_attributes)
                .where("LENGTH(reply_timestamp)>0 OR LENGTH(retweet_timestamp)>0 OR "
                       "LENGTH(retweet_with_comment_timestamp)>0 OR LENGTH(like_timestamp)>0")
        )

        # Negative samples (no interactions)
        sdf_no_interactions = (
            sdf_raw_data.selectExpr(*engagements_attributes)
                .where("LENGTH(reply_timestamp)=0 AND LENGTH(retweet_timestamp)=0 AND "
                       "LENGTH(retweet_with_comment_timestamp)=0 AND LENGTH(like_timestamp)=0")
        )

        # Convert to koalas DataFrame
        ks_users = sdf_users.to_koalas()
        ks_interactions = sdf_interactions.to_koalas()
        ks_no_interactions = sdf_no_interactions.to_koalas()

    else:  # Inference time: targets are unavailable, raw_data is test data
        # We do not extend anything. XGBoost will handle sparsity for us
        ks_users_train = auxiliary_train["engagement_graph_vertices"]
        ks_interactions_train = auxiliary_train["engagement_graph_edges_interactions"]
        ks_no_interactions_train = auxiliary_train["engagement_graph_edges_no_interactions"]

        ks_users = ks_users_train
        ks_interactions = ks_interactions_train
        ks_no_interactions = ks_no_interactions_train

    return {
        "engagement_graph_vertices": ks_users,
        "engagement_graph_edges_interactions": ks_interactions,
        "engagement_graph_edges_no_interactions": ks_no_interactions,
    }
