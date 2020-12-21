def build_ALS(X_train, rank=10, maxIter=10, regParam=0.1):
    from pyspark.ml.recommendation import ALS
    als = ALS(rank=rank, maxIter=maxIter, regParam=regParam, seed=42)
    train = X_train.select(['user_id', 'book_id', 'rating']).toDF('user', 'item', 'rating')
    model = als.fit(train)
    return model
