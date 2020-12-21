#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct
from pyspark import SparkContext as sc

def main(spark, file_path, subsampling=1):
    """
    This function splits a dataframe into the train/valid/test set.
    
    - train: randomly sample 60% of users and include all of their interactions
                + 50% of interactions from users in the valid/test set
    - valid: randomly sample 20% of users and include 50% of their interactions
    - test : randomly sample 20% of users and include 50% of their interactions
    
    Random sampling of users and interactions results in mutually exclusive splits.
    Parameters
    ----------
    spark : spark session object
    file_path : string; The path (in HDFS) to the CSV file, e.g., `hdfs:/user/bm106/pub/people_small.csv`
    subsampling: subsample 1% if True
    ----------
    """

    # Load the CSV/parquet file and Set the column name in case it's missing
    df = spark.read.parquet(file_path)
    ##### df = df.toDF('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed')  

    # Create a single-column dataframe with distinct user_ids and Randomly split into train/valid/test user groups
    user_list = df.select("user_id").distinct()
    
    # Subsample if true
    print("start subsampling:", subsampling)
    user_list = user_list.sample(False, fraction=subsampling, seed=42)
    
    user_train, user_valid, user_test = user_list.randomSplit([0.6, 0.2, 0.2], seed = 42)

    # Create X_train
    df.createOrReplaceTempView('df')
    user_train.createOrReplaceTempView('user_train')
    X_train = spark.sql('SELECT * FROM df WHERE user_id IN (SELECT user_id FROM user_train)')

    # Create X_valid
    user_valid.createOrReplaceTempView('user_valid')
    X_valid = spark.sql('SELECT * FROM df WHERE user_id IN (SELECT user_id FROM user_valid)')
    X_valid_sampled = X_valid.sampleBy("user_id", fractions={k['user_id']: 0.5 for k in user_valid.rdd.collect()}, seed=42)
    X_valid_to_train = X_valid.subtract(X_valid_sampled)  # This dataframe will be concatenated with X_train

    # Create X_test
    user_test.createOrReplaceTempView('user_test')
    X_test = spark.sql('SELECT * FROM df WHERE user_id IN (SELECT user_id FROM user_test)')
    X_test_sampled = X_test.sampleBy("user_id", fractions={k['user_id']: 0.5 for k in user_test.rdd.collect()}, seed=42)
    X_test_to_train = X_test.subtract(X_test_sampled)

    # Concatenate remaining records of valid/test to X_train
    X_train = X_train.union(X_valid_to_train).union(X_test_to_train)

    return X_train, X_valid_sampled, X_test_sampled

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('data_split').getOrCreate()

    # Get the filename from the command line
    file_path = sys.argv[1]

    # Call our main routine
    main(spark, file_path) 
