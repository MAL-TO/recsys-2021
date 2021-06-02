import databricks.koalas as ks

# Hour of day is not meaningful due to time zones
# What can be useful instead is hour_of_day targed encoded with language

def te_language_hour(raw_data, features = None, auxiliary_dict = None):
    """
    Target encoding grouping by language and hour of the day
    """
    ALPHA=20 # smoothing factor
    targets = ["reply", "retweet", "retweet_with_comment", "like"]
    target_encoded = {}
    for target in targets:
        target_series = features[target].apply(lambda x: 1 if x else 0)
        te_df = ks.concat([raw_data, features['hour_of_day'], target_series], axis=1, join="inner").reset_index(drop=False)
        te_series = ks.sql(f'''SELECT mean({target}) as te_{target}, count(*) as num_samples, hour_of_day, language
        FROM {{te_df}} 
        GROUP BY hour_of_day, language''')
    

        te_df.merge(te_series, how = 'left', on=['hour_of_day', 'language'])
        print(te_series.head())
        print(type(te_series))
        # print(te_series.head())
        print(te_series.describe())
        print()

    # target_encoded = df.groupBy(conditioning_rv).\
    # agg({f"{target}":"mean", "*":"count"}).\
    # withColumnRenamed(f"avg({target})", f"avg_{target_name}").\
    # withColumnRenamed("count(1)", "num_samples")
    
    # # Computed the smoothed mean: (n*x_avg + alpha*target_avg) / (n + alpha)
    # target_encoded = target_encoded.withColumn(f"smoothed_avg_{target_name}",
    #                           (col("num_samples")*col(f"avg_{target_name}") + (alpha*target_mean)) / \
    #                           (col("num_samples") + alpha))

