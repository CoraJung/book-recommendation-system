def get_rankMetrics(spark, df, trained_model, approx=False, k=500):
    """
    This function evaluates the performance of a given model on a given dataset using Ranking Metrics,
    and returns the final performance metrics.

    Parameters
    ----------
    df: DataFrame to evaluate on
    trained_model: trained model to evaluate
    approx: boolean; use ANN(approximate nearest neighbors) when True
    k: number of recommendation 
    ----------
    """
    import datetime
    import nmslib_recommend2
    import pyspark.sql.functions as F
    from pyspark.mllib.evaluation import RankingMetrics
    
    # change column names
    df = df.select(['user_id', 'book_id', 'rating']).toDF('user', 'item', 'rating')
    
    # relevant item if its centered rating > 0
    fn = F.udf(lambda x: 1.0 if x >= 3 else 0.0)
    df = df.withColumn('rating', fn(df.rating))
    relevant = df[df.rating == 1.0].groupBy('user').agg(F.collect_list('item'))
    
    # recommend k items for each user
    print("recommendation time comparison start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if approx:
        recommend = nmslib_recommend2.nmslib_recommend(spark, df, trained_model, k)
        recommend = spark.createDataFrame(recommend , ["user", "recommend"])
        joined = recommend.join(relevant, on='user')
        rec_and_rel = []
        for user, rec, rel in joined.collect():
            rec_and_rel.append((rec, rel))
    else:     
        userSubset = relevant.select('user')
        recommend = trained_model.recommendForUserSubset(userSubset, 500)
        joined = recommend.join(relevant, on='user')
        rec_and_rel = []
        for user, rec, rel in joined.collect():
            predict_items = [i.item for i in rec]
            rec_and_rel.append((predict_items, rel))
    print("recommendation time comparison end: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Compute metrics
    rec_and_rel_rdd = spark.sparkContext.parallelize(rec_and_rel)
    metric_class = RankingMetrics(rec_and_rel_rdd)

    ndcg = metric_class.ndcgAt(k)
    map_ = metric_class.meanAveragePrecision
    pk = metric_class.precisionAt(k)

    return print("NDCG:", ndcg, "\nMAP:", map_, "\nPrecision:", pk)
