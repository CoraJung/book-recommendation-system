# BIG DATA
## Final project

*Submission deadline*: 2020-05-11


# Overview

In the final project, we built and evaluated a recommender system. It is intended more as an opportunity to integrate multiple techniques to solve a realistic, large-scale applied problem.

## The data set

In this project, we used the [Goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) collected by 
> Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", RecSys 2018.

  - `goodreads_interactions.csv`
  - `user_id_map.csv`
  - `book_id_map.csv`

The first file contains tuples of user-book interactions.  For example, the first five lines are
```
user_id,book_id,is_read,rating,is_reviewed
0,948,1,5,0
0,947,1,5,1
0,946,1,5,0
0,945,1,5,0
```

The other two files consist of mappings between the user and book numerical identifiers used in the interactions file, and their alphanumeric strings which are used in supplementary data (see below).
Overall there are 876K users, 2.4M books, and 223M interactions.

## Basic recommender system 

My recommendation model used Spark's alternating least squares (ALS) method to learn latent factor representations for users and items. [pyspark.ml.recommendation module](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.recommendation)

This model has some hyper-parameters that I tuned to optimize performance on the validation set, notably: 

  - the *rank* (dimension) of the latent factors, and
  - the regularization parameter *lambda*.

### Data splitting and subsampling

First, I constructed train, validation, and test splits of the data using a fixed random seed and save the results, so that validation scores are comparable across runs.

Data splitting for recommender system interactions (user-item ratings) can be a bit more delicate than the typical randomized partitioning that you might encounter in a standard regression or classification setup. 

As a general recipe, we did following:
  - Select 60% of users (and all of their interactions) to form the *training set*.
  - Select 20% of users to form the *validation set*.  For each validation user, use half of their interactions for training, and the other half should be held out for validation.  (Remember: you can't predict items for a user with no history at all!)
  - Remaining users: same process as for validation.

As mentioned below, we downsampled the data when prototyping my implementation.  
Downsampling should follow similar logic to partitioning: don't downsample interactions directly.
Instead, we sampled a percentage of users, and take all of their interactions to make a miniature version of the data.

Any items not observed during training (i.e., which have no interactions in the training set, or in the observed portion of the validation and test users), were omitted (unless we're implementing cold-start recommendation as an extension).

In general, users with few interactions (say, fewer than 10) may not provide sufficient data for evaluation, especially after partitioning their observations into train/test. We discarded these users from the experiment.

### Evaluation

Once our model was trained, we evaluated its accuracy on the validation and test data.
Scores for validation and test are both reported in our final writeup. Evaluations were based on predicted top 500 items for each user.

The choice of evaluation criteria for hyper-parameter tuning is up to you, as is the range of hyper-parameters you consider, but be sure to document your choices in the final report.
As a general rule, you should explore ranges of each hyper-parameter that are sufficiently large to produce observable differences in your evaluation score.

In addition to the RMS error metric, Spark provides some additional evaluation metrics which you can use to evaluate your implementation.
Refer to the [ranking metrics](https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#ranking-systems) section of the documentation for more details.
If you like, you may also use additional software implementations of recommendation or ranking metric evaluations, but please cite any additional software you use in the project.

### Results

The best hyper-parameter setting against the validation set is:
rank = 20, regularization parameter = 0.05 with NDCG score: 0.01255, MAP: 0.00038, precision at K = 0.00307.

The performance on test set is as following:
**NDCG**:	0.0129
**MAP**: 0.0005
**Precision at 500**: 0.0032

Our model with the best hyper-parameters seems well generalized as it can be seen from the similar performance on validation and test set.

### Hints
We first started small, and got the entire system working start-to-finish before investing time in hyper-parameter tuning!
We also found it helpful to convert the raw CSV data to parquet format for more efficient access.
We also downsampled the data to more rapidly prototype our model and we made sure that our downsampled data includes enough users from the validation set to test the model.
We suggest building sub-samples of 1%, 5%, and 25% of the data, and then running the entire set of experiments end-to-end on each sample before attempting the entire dataset.


## Extensions
We also implemented a fast search extension on top of the baseline collaborative filter model.
  - *Fast search*: use a spatial data structure (e.g., LSH or partition trees) to implement accelerated search at query time.  For this, it is best to use an existing library such as [annoy](https://github.com/spotify/annoy) or [nmslib](https://github.com/nmslib/nmslib), and you will need to export the model parameters from Spark to work in your chosen environment.  For full credit, you should provide a thorough evaluation of the efficiency gains provided by your spatial data structure over a brute-force search method.

# What this repo has
This repo contains all of our code, a final report describing our implementation, evaluation results, and extensions.
