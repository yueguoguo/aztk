# Load modules.

from __future__ import print_function

import json
import os
import sys

import tempfile
import time
import shutil
import math

import pyspark
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics 
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel 
from pyspark.mllib.recommendation import Rating 
from pyspark.sql import Window
from pyspark.sql.functions import col, expr
import pyspark.sql.functions as F

import numpy as np
import pandas as pd

# Initialize a Spark session.

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import SparkSession
except ImportError as e:
    print ("Error importing Spark Modules", e)
    sys.exit(1)
try:
    conf = SparkConf()
    conf.set("spark.executor.memory", "12g")
    conf.set("spark.driver.memory", "12g")
    conf.set("spark.cores.max", "20")
    conf.set("spark.network.timeout", 1000000)
    conf.setAppName("recommendpy")
    spark = pyspark.sql.SparkSession.builder.appName("PySparkRec").config(conf=conf).getOrCreate()
    print("Spark Version Required >2.1; actual: "+str(spark.version))
    sc=spark.sparkContext
    # sc.setCheckpointDir("checkpoint/")
except ImportError as e:
    print("Error initializing Spark", e)
    sys.exit(1)

# Access data files vis WASB protocol.

wasb_prefix = "wasb://recsys@zhledata.blob.core.windows.net"
# file_folder = "movielens"
file_folder = "movielens100k"

# url_rating, url_movie, url_catalog = [wasb_prefix + "/" + file_folder + "/" + s for s in ["ratings.csv", "movie.csv", "catalog.csv"]]
url_rating, url_movie, url_catalog = [wasb_prefix + "/" + file_folder + "/" + s for s in ["ratings_api_train_small.csv", "movie.csv", "catalog.csv"]]

# Functions for use.

def rating_info(ratings):
    nUsers=ratings.select('customerID').distinct().count()
    itemCatalog = ratings.select('itemID').distinct()
    nItems=itemCatalog.count()
    nRatings = ratings.count()

    print("\nTotal n. of users: "+str(nUsers))
    print("\nTotal n. of items: "+str(nItems))
    print("Total n. of ratings: "+str(nRatings))

def thresholding(ratings, threshold):
    tmpDF = ratings.groupBy('customerID').agg({"itemID": "count"}).withColumnRenamed(
        'count(itemID)', 'nitems').where(col('nitems') >= threshold)

    tmpDF2 = ratings.groupBy('itemID').agg({"customerID": "count"}).withColumnRenamed(
        'count(customerID)', 'ncustomers').where(col('ncustomers') >= threshold)

    inputDF = tmpDF2.join(ratings, 'itemID').drop('ncustomers').join(tmpDF, 'customerID').drop('nitems') 

    print("Ratings: " + str(inputDF.count()))

    return(inputDF)

def train_test_split(ratings, ratio):
    # Stratified sampling by item into a train and test data set

    nusers_by_item = ratings.groupBy('itemID').agg({"customerID": "count"}).withColumnRenamed('count(customerID)', 'nusers').rdd

    perm_indices = nusers_by_item.map(lambda r: (r[0], np.random.permutation(r[1]), r[1]))

    tr_idx = perm_indices.map(lambda r: (r[0], r[1][: int(round(r[2] * ratio))]))
    test_idx = perm_indices.map(lambda r: (r[0], r[1][int(round(r[2] * ratio)):]))

    tr_by_item = ratings.rdd \
                    .groupBy(lambda r: r[1]) \
                    .join(tr_idx).flatMap(lambda col: np.array([x for x in col[1][0]])[col[1][1]])
    test_by_item = ratings.rdd \
                    .groupBy(lambda r: r[1]) \
                    .join(test_idx).flatMap(lambda col: np.array([x for x in col[1][0]])[col[1][1]])

    train = tr_by_item.map(lambda r: tuple(r))
    test = test_by_item.map(lambda r: tuple(r))

    print("Training ratings: " + str(train.count()),
          "\nTesting ratings: " + str(test.count()))

    return train, test

def train_als(ratings, explicit, rank, rp, iteration, non_negative):

    # To avoid stackoverflow issue.

    sc.setCheckpointDir("checkpoint/")
    ALS.checkpointInterval = 2

    if explicit == True:
        model = ALS.train(ratings, rank=rank, iterations=iteration, lambda_=rp, nonnegative=non_negative)
    else:
        model = ALS.trainImplicit(ratings, rank=r, iterations=iteration, lambda_=rp, nonnegative=non_negative)

    return model

class Recommendation:
    """Class for recommendation"""

    def __init__(self, ratings, model, k):
        self.ratings = ratings
        self.model = model
        self.k = k

    def actual_recommendations(self):
        windowSpec = Window.partitionBy('customerID').orderBy(col('rating').desc())
        perUserActualItemsDF = self.ratings \
            .select('customerID', 'itemID', 'rating', F.rank().over(windowSpec).alias('rank')) \
            .where('rank <= {0}'.format(self.k)) \
            .groupBy('customerID') \
            .agg(expr('collect_list(itemID) as actual'))

        return perUserActualItemsDF

    def recommendation(self):
        userRecsRDD = self.model.recommendProductsForUsers(self.k)

        userRecsDF=spark.createDataFrame(
            # userRecsRDD.flatMap(lambda r: r[1]).repartition(10)
            userRecsRDD.flatMap(lambda r: r[1])
        )

        perUserRecommendedItemsDF=userRecsDF.select("user", "product") \
            .withColumnRenamed('user', 'customerID') \
            .groupBy('customerID').agg(expr('collect_list(product) as recommended'))

        perUserActualItemsDF = self.actual_recommendations()

        joined_rec_actual = perUserRecommendedItemsDF.join(perUserActualItemsDF, on='customerID').drop('customerID')

        return joined_rec_actual

    def max_diversity(self):
        perUserActualItemsDF = self.actual_recommendations()

        nItems = self.ratings.map(lambda r: r[1]).distinct().count()

        uniqueItemsActual = perUserActualItemsDF.rdd.map(lambda row: row[1]).reduce(lambda x,y: set(x).union(set(y)))
        maxDiversityAtk = len(uniqueItemsActual) / nItems

        return maxDiversityAtk

    def get_metrics(self):
        items_recommendation = self.recommendation()

        items_recommendation.show()

        # metrics = RankingMetrics(items_recommendation.rdd)

        # k = int(self.k)

        # patk = metrics.precisionAt(k)

        # ndcg = metrics.ndcgAt(k)

        # mavgp = metrics.meanAveragePrecision(k)

        recall = items_recommendation.rdd.map(lambda x: len(set(x[0]).intersection(set(x[1])))/len(x[1])).mean()

        nItems = self.ratings.rdd.map(lambda r: r[1]).distinct().count()
        uniqueItemsRecommended = items_recommendation.rdd.map(lambda row: row[0]) \
                .reduce(lambda x,y: set(x).union(set(y)))
        diversity = len(uniqueItemsRecommended) / nItems

        # return patk, ndcg, mavgp, recall, diversity
        return recall, diversity
    
explicit = True

if explicit == True:
    ratings = spark.read.csv(url_rating, header='false').drop_duplicates()

    ratings = ratings.withColumnRenamed("_c0", "customerID") \
                .withColumnRenamed("_c1", "itemID") \
                .withColumnRenamed("_c2", "rating") \
                .drop("_c3") \
                .drop_duplicates()
else:
    print(" ")

rating_info(ratings)

# Parameters used in the experimentation.

threshold = 0
split_ratio = 0.65
r = 80
rp = 0.01
k = 10

# Step1 - process the data.

time_start_processing = time.time()

ratings = thresholding(ratings, threshold=threshold)

train, test = train_test_split(ratings, ratio=split_ratio)

train.cache()
test.cache()

print("It takes {0} to finish processing".format(time.time() - time_start_processing))

# Step 2 - train ALS model.

time_start_training = time.time()

# To avoid stackoverflow issue.

sc.setCheckpointDir("checkpoint/")
# ALS.checkpointInterval = 2

# als_model = train_als(ratings=train, explicit=True, rank=r, rp=rp, iteration=30, non_negative=True)
als_model = ALS.train(ratings=train, rank=r, iterations=30, lambda_=rp, nonnegative=True)

print("It takes {0} to finish training".format(time.time() - time_start_training))

# Step 3 - evaluate ALS model.

test_ratings = test.map(lambda r: (int(r[0]), int(r[1]), int(r[2]))) \
                .toDF(schema=['customerID', 'itemID', 'rating'])

time_start_evaluating = time.time()

recommender = Recommendation(ratings=test_ratings, model=als_model, k=k)
# patk, ndcg, mavgp, recall, diversity = recommender.get_metrics()
recall, diversity = recommender.get_metrics()

# print([patk, ndcg, mavgp, recall, diversity])
print([recall, diversity])

print("It takes {0} to finish evaluating".format(time.time() - time_start_evaluating))

# Stop Spark session.

print("All work well!")

spark.stop()
