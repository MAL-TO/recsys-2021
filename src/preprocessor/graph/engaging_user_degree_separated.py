import graphframes


def engaging_user_degree(raw_data, features, auxiliaries):
    # Get vertices and edges of the engagement graphs
    # sdf_vertices schema: ["id"]
    # sdf_edges_dict[any]: ["src", "dst"]
    sdf_vertices = auxiliaries["engagement_graph_vertices"].to_spark()
    sdf_edges_dict = {
        "reply": auxiliaries["engagement_graph_edges_reply"].to_spark(),
        "retweet": auxiliaries["engagement_graph_edges_retweet"].to_spark(),
        "retweet_with_comment": auxiliaries["engagement_graph_edges_retweet_with_comment"].to_spark(),
        "like": auxiliaries["engagement_graph_edges_like"].to_spark(),
    }

    # Extract index from raw data
    sdf_raw_data = raw_data.reset_index(drop=False).to_spark()
    sdf_raw_data_index = sdf_raw_data.selectExpr("tweet_id", "engaging_user_id AS id")

    ks_engaging_user_in_degree = {}
    ks_engaging_user_out_degree = {}

    # Create different graphs for different types of interactions
    for typ in sdf_edges_dict:
        g = graphframes.GraphFrame(sdf_vertices, sdf_edges_dict[typ])

        # Extract degrees of users (returns DataFrame with cols
        # "id" and "{in,out}Degree"
        # Note: vertices with 0 in-edges are not returned in the result
        sdf_user_in_degree = g.inDegrees
        sdf_user_out_degree = g.outDegrees

        # Fill degrees with vertices having 0 in-edges
        sdf_all_user_in_degree = sdf_vertices.join(sdf_user_in_degree, on="id", how="left_outer")
        sdf_all_user_in_degree = sdf_all_user_in_degree.fillna(0)
        sdf_all_user_out_degree = sdf_vertices.join(sdf_user_out_degree, on="id", how="left_outer")
        sdf_all_user_out_degree = sdf_all_user_out_degree.fillna(0)

        # Join to index
        sdf_engaging_user_in_degree = sdf_raw_data_index.join(sdf_all_user_in_degree, on="id", how="left_outer")
        sdf_engaging_user_out_degree = sdf_raw_data_index.join(sdf_all_user_out_degree, on="id", how="left_outer")

        # Create feature
        kdf_engaging_user_in_degree = sdf_engaging_user_in_degree.withColumnRenamed("id", "engaging_user_id").to_koalas()
        kdf_engaging_user_out_degree = sdf_engaging_user_out_degree.withColumnRenamed("id", "engaging_user_id").to_koalas()

        ks_engaging_user_in_degree[typ] = (
            kdf_engaging_user_in_degree
            .set_index(["tweet_id", "engaging_user_id"])
            ["inDegree"]
            .rename(f"inDegree_{typ}")
        )
        ks_engaging_user_out_degree[typ] = (
            kdf_engaging_user_out_degree
            .set_index(["tweet_id", "engaging_user_id"])
            ["outDegree"]
            .rename(f"outDegree_{typ}")
        )

    return {
        "engaging_user_in_degree_reply": ks_engaging_user_in_degree["reply"],
        "engaging_user_out_degree_reply": ks_engaging_user_out_degree["reply"],
        "engaging_user_in_degree_retweet": ks_engaging_user_in_degree["retweet"],
        "engaging_user_out_degree_retweet": ks_engaging_user_out_degree["retweet"],
        "engaging_user_in_degree_retweet_with_comment": ks_engaging_user_in_degree["retweet_with_comment"],
        "engaging_user_out_degree_retweet_with_comment": ks_engaging_user_out_degree["retweet_with_comment"],
        "engaging_user_in_degree_like": ks_engaging_user_in_degree["like"],
        "engaging_user_out_degree_like": ks_engaging_user_out_degree["like"],
    }
