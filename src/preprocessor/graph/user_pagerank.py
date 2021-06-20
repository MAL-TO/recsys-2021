import graphframes

def user_pagerank(raw_data, features, auxiliaries):
    index_col = ["tweet_id", "engaging_user_id"]

    # Extract index from raw data
    sdf_raw_data = raw_data.to_spark(index_col=index_col)

    # Get vertices and edges of the engagement graphs
    # sdf_vertices schema: ["id"]
    # sdf_edges_dict[any]: ["src", "dst"]
    sdf_vertices = auxiliaries["engagement_graph_vertices"].to_spark()
    sdf_edges = auxiliaries["engagement_graph_edges_interactions"].to_spark()
    g = graphframes.GraphFrame(sdf_vertices, sdf_edges)

    # Compute PageRank
    gpr = g.pageRank(resetProbability=0.15, maxIter=300, tol=None)

    # Returns DataFrame with cols "id" and "pagerank"
    sdf_user_pagerank = gpr.vertices

    # Join to index
    sdf_index_engaging_user = sdf_raw_data.selectExpr("tweet_id", "engaging_user_id AS id")
    sdf_engaging_user_pagerank = sdf_index_engaging_user.join(sdf_user_pagerank, on="id", how="left_outer")

    sdf_index_engaged_with_user = sdf_raw_data.selectExpr("tweet_id", "engaging_user_id", "engaged_with_user_id AS id")
    sdf_engaged_with_user_pagerank = sdf_index_engaged_with_user.join(sdf_user_pagerank, on="id", how="left_outer")

    # Create features
    kdf_engaging_user_pagerank = (
        sdf_engaging_user_pagerank
        .withColumnRenamed("id", "engaging_user_id")
        .to_koalas()
        .set_index(index_col)
        ["pagerank"]
        .rename("engaging_user_pagerank")
    )
    kdf_engaged_with_user_pagerank = (
        sdf_engaged_with_user_pagerank
        .to_koalas()
        .set_index(index_col)
        ["pagerank"]
        .rename("engaged_with_user_pagerank")
    )

    return {
        "engaging_user_pagerank": kdf_engaging_user_pagerank,
        "engaged_with_user_pagerank": kdf_engaged_with_user_pagerank,
    }