import graphframes


def engaging_user_degree(raw_data, features, auxiliaries):
    sdf_vertices = auxiliaries["engagement_graph_users"].to_spark()
    sdf_edges = auxiliaries["engagement_graph_engagements"].to_spark()

    g = graphframes.GraphFrame(sdf_vertices, sdf_edges)

    # Extract degrees of users (nodes)
    # Note: vertices with 0 in-edges are not returned in the result
    sdf_user_in_degree = g.inDegrees
    sdf_user_out_degree = g.outDegrees

    sdf_raw_data = raw_data.reset_index(drop=False).to_spark()
    sdf_raw_data_index = sdf_raw_data.selectExpr("tweet_id", "engaging_user_id AS id")

    sdf_engaging_user_in_degree = sdf_raw_data_index.join(
        sdf_user_in_degree, on="id", how="left_outer"
    )
    sdf_engaging_user_out_degree = sdf_raw_data_index.join(
        sdf_user_out_degree, on="id", how="left_outer"
    )

    # Fill `None`s with zeros, as `None` corresponds to degree 0
    sdf_engaging_user_in_degree = sdf_engaging_user_in_degree.fillna(0)
    sdf_engaging_user_out_degree = sdf_engaging_user_out_degree.fillna(0)

    kdf_engaging_user_in_degree = sdf_engaging_user_in_degree.withColumnRenamed(
        "id", "engaging_user_id"
    ).to_koalas()
    kdf_engaging_user_out_degree = sdf_engaging_user_out_degree.withColumnRenamed(
        "id", "engaging_user_id"
    ).to_koalas()

    ks_engaging_user_in_degree_with_index = kdf_engaging_user_in_degree.set_index(
        ["tweet_id", "engaging_user_id"]
    )["inDegree"]
    ks_engaging_user_out_degree_with_index = kdf_engaging_user_out_degree.set_index(
        ["tweet_id", "engaging_user_id"]
    )["outDegree"]

    return {
        "engaging_user_in_degree": ks_engaging_user_in_degree_with_index,
        "engaging_user_out_degree": ks_engaging_user_out_degree_with_index,
    }
