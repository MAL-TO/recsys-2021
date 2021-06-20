import graphframes
from pyspark.ml.feature import QuantileDiscretizer


def engaging_user_interaction_degree(raw_data, features, auxiliaries):
    index_col = ["tweet_id", "engaging_user_id"]

    # Extract index from raw data
    sdf_raw_data = raw_data.to_spark(index_col=index_col)
    sdf_raw_data_index = sdf_raw_data.selectExpr("tweet_id", "engaging_user_id AS id")

    # Get vertices and edges of the engagement graphs
    # sdf_vertices schema: ["id"]
    # sdf_edges_dict[any]: ["src", "dst"]
    sdf_vertices = auxiliaries["engagement_graph_vertices"].to_spark()
    sdf_edges = auxiliaries["engagement_graph_edges_interactions"].to_spark()
    g = graphframes.GraphFrame(sdf_vertices, sdf_edges)

    # Extract degrees of users
    # Returns DataFrame with cols "id" and "{in,out}Degree"
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

    qds = QuantileDiscretizer(numBuckets=5, relativeError=0.01, handleInvalid="error")
    qds.setInputCol("inDegree")
    qds.setOutputCol("inDegreeQuantized")
    sdf_engaging_user_in_degree = qds.fit(sdf_engaging_user_in_degree).transform(sdf_engaging_user_in_degree)
    qds.setInputCol("outDegree")
    qds.setOutputCol("outDegreeQuantized")
    sdf_engaging_user_out_degree = qds.fit(sdf_engaging_user_out_degree).transform(sdf_engaging_user_out_degree)

    # Create feature
    kdf_engaging_user_in_degree = sdf_engaging_user_in_degree.withColumnRenamed("id", "engaging_user_id").to_koalas()
    kdf_engaging_user_out_degree = sdf_engaging_user_out_degree.withColumnRenamed("id", "engaging_user_id").to_koalas()

    ks_engaging_user_interaction_in_degree = (
        kdf_engaging_user_in_degree
        .set_index(index_col)
        ["inDegreeQuantized"]
        .rename("engaging_user_interaction_in_degree")
        .astype("float")
    )
    ks_engaging_user_interaction_out_degree = (
        kdf_engaging_user_out_degree
        .set_index(index_col)
        ["outDegreeQuantized"]
        .rename("engaging_user_interaction_out_degree")
        .astype("float")
    )

    return {
        "engaging_user_interaction_in_degree": ks_engaging_user_interaction_in_degree,
        "engaging_user_interaction_out_degree": ks_engaging_user_interaction_out_degree,
    }
