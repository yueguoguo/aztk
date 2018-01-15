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
# NOTE: not sure whether this is the right way of configuring Spark session with aztk. 

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import SparkSession
except ImportError as e:
    print ("Error importing Spark Modules", e)
    sys.exit(1)
try:
    conf = SparkConf()
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "3g")
    conf.set("spark.cores.max", "16")
    conf.set("spark.network.timeout", 1000000)
    conf.setAppName("recommendpy")
    spark = pyspark.sql.SparkSession.builder.appName("recpy1").config(conf=conf).getOrCreate()
    print("Spark Version Required >2.1; actual: "+str(spark.version))
    sc=spark.sparkContext
except ImportError as e:
    print("Error initializing Spark", e)
    sys.exit(1)

# Access data files vis WASB protocol.

wasb_prefix = "wasb://recsys@zhledata.blob.core.windows.net"
file_folder = "movielens"

url_rating, url_movie, url_catalog = [wasb_prefix + "/" + file_folder + "/" + s for s in ["ratings.csv", "movie.csv", "catalog.csv"]]

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

    tr_by_item = ratings.rdd.groupBy(lambda r: r[1]).join(tr_idx).flatMap( lambda r: np.array([x for x in r[1][0]]) [r[1][1]] )
    test_by_item = ratings.rdd.groupBy(lambda r: r[1]).join(test_idx).flatMap( lambda r: np.array([x for x in r[1][0]]) [r[1][1]] )

    train = tr_by_item.map(lambda r: tuple(r))
    test = test_by_item.map(lambda r: tuple(r))

    print("Training ratings: " + str(train.count()),
          "\nTesting ratings: " + str(test.count()))

    return train, test

def train_als(ratings, explicit, rank, rp, iteration, non_negative):
    if explicit == True:
        model = ALS.train(ratings, rank=r, iterations=30, lambda_=rp, nonnegative=True)
    else:
        model = ALS.trainImplicit(ratings, rank=r, iterations=30, lambda_=rp, nonnegative=True)

    return model

def actual_recommendations(ratings): 
    windowSpec = Window.partitionBy('customerID').orderBy(col('rating').desc())
    perUserActualItemsDF = test_ratings \
        .select('customerID', 'itemID', 'rating', F.rank().over(windowSpec).alias('rank')) \
        .where('rank <= {0}'.format(k)) \
        .groupBy('customerID') \
        .agg(expr('collect_list(itemID) as actual'))

    return perUserActualItemsDF

def max_diversity(ratings):
    perUserActualItemsDF = actual_recommendations(ratings)

    nItems = test_ratings.map(lambda r: r[1]).distinct().count()

    uniqueItemsActual = perUserActualItemsDF.rdd.map(lambda row: row[1]).reduce(lambda x,y: set(x).union(set(y)))
    maxDiversityAtk = len(uniqueItemsActual) / nItems

    return maxDiversityAtk

def recommendation(ratings, model, k):
    userRecsRDD = model.recommendProductsForUsers(k)

    userRecsDF=spark.createDataFrame(
        userRecsRDD.flatMap(lambda r: r[1]).repartition(10)
    )

    perUserRecommendedItemsDF=userRecsDF.select("user", "product") \
        .withColumnRenamed('user', 'customerID') \
        .groupBy('customerID').agg(expr('collect_list(product) as recommended'))

    perUserActualItemsDF = actual_recommendations(ratings)

    joined_rec_actual = perUserRecommendedItemsDF.join(perUserActualItemsDF, on='customerID').drop('customerID')

    return joined_rec_actual

def get_metrics(recommendations):
    metrics = RankingMetrics(recommendations.rdd)

    return metrics

def patk(metrics, k):
    patk = metrics.precisionAt(k)

    return patk

def ndcgatk(metrics, k):
    ndcg = metrics.ndcgAt(k)

    return ndcg

def mapatk(metrics, k):
    map = metrics.meanAveragePrecision(k)

    return map

def recallatk(recommendations):
    recall = recommendations.rdd.map(lambda x: len(set(x[0]).intersection(set(x[1])))/len(x[1])).mean()

    return recall

def diversityatk(recommendations):
    nItems = test_ratings.map(lambda r: r[1]).distinct().count()

    uniqueItemsRecommended = recommendations.rdd.map(lambda row: row[0]) \
            .reduce(lambda x,y: set(x).union(set(y)))

    diversity = len(uniqueItemsRecommended) / nItems

    return diversity
    
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

# Step1 - process the data.

time_start_processing = time.time()

ratings = thresholding(ratings, threshold = 0)

train, test = train_test_split(ratings, ratio = 0.65)

print("It takes {0} to finish processing".format(time.time() - time_start_processing))

train.cache()
test.cache()

# Step 2 - train ALS model.

time_start_training = time.time()

r = 80
rp = 0.01

model = train_als(train, explicit, r, rp, 30, True)

print("It takes {0} to finish training".format(time.time() - time_start_training))

# Step 3 - evaluate ALS model.

test_ratings = test.toDF(schema=['customerID', 'itemID', 'rating'])

time_start_evaluating = time.time()

k = 5

print("It takes {0} to finish evaluating".format(time.time() - time_start_evaluating))

# Stop Spark session.

print("All work well!")

spark.stop()