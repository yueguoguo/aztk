# Training recommender.

import numpy as np
import pandas as pd
import sys
import time

import pyspark
from pyspark.mllib.recommendation import ALS

# Initialize Spark session.

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import SparkSession
except ImportError as e:
    print ("Error importing Spark Modules", e)
    sys.exit(1)
try:
    conf = SparkConf()
    conf.setAppName("RecommendPy")
    spark = pyspark.sql.SparkSession.builder.config(conf=conf).getOrCreate()
    print("Spark Version Required >2.1; actual: "+str(spark.version))
    sc=spark.sparkContext
except ImportError as e:
    print("Error initializing Spark", e)
    sys.exit(1)

# Simulate a list of movie preference.
# TODO: eventually this list will be streamed from aipk arguments or a csv if the demo works in batch mode.

time_start = time.time()

movie_list = [
    "Toy Story (1995)", 
    "Jumanji (1995)",
    "Grumpier Old Men (1995)",
    "Waiting to Exhale (1995)",
    "Father of the Bride Part II (1995)"
]

url_movie = "https://zhledata.blob.core.windows.net/recsys/movielens/movies.csv"
url_rating = "https://zhledata.blob.core.windows.net/recsys/movielens100k.csv"

df_movie = pd.read_csv(url_movie)
df_movie = df_movie.iloc[:,0:2]
df_movie.columns = ['item', 'title']

# df_rating = pd.read_csv(url_rating, header=None, sep="\t")
df_rating = pd.read_csv(url_rating)
df_rating = df_rating.iloc[:,0:3]
df_rating.columns = ['user', 'item', 'rating']

df_rating_preference = df_movie[df_movie['title'].isin(movie_list)]

# Arrange input data as user rating data frame and append it into other rating records.
# NOTE: we assume explicit ratings for this case, and the input movies be default have ratings of 5.

user_id = df_rating.user.max().item() + 1
count_of_ratings = df_rating_preference.shape[0]

df_rating_preference['rating'] = pd.Series(np.repeat(5, count_of_ratings), index=df_rating_preference.index)
df_rating_preference['user'] = pd.Series(np.repeat(user_id, count_of_ratings), index=df_rating_preference.index)
df_rating_preference['item'] = df_rating_preference['item'].astype(int)
df_rating_preference = df_rating_preference[['user', 'item', 'rating']]

df_rating = df_rating.append(df_rating_preference)

# Read rating data into Spark DataFrame.

dfs_rating = spark.createDataFrame(df_rating)    

# Train an ALS model.

rdd_rating = dfs_rating.rdd
rdd_rating.cache()

# sc.setCheckpointDir('./checkpoint/')
ALS.checkpointInterval = 2

model=ALS.train(
    ratings= rdd_rating, 
    rank=10, 
    iterations=10, 
    lambda_=0.01, 
    nonnegative=True
)

# Recommend 10 moviews to a user.

k = 10
        
rdd_rec = model.recommendProductsForUsers(k)
dfs_rec = spark.createDataFrame(rdd_rec.flatMap(lambda r: r[1]))

df_user_rec = dfs_rec.filter(dfs_rec.user == user_id).toPandas()
df_user_rec.columns = ["user", "item", "rating"]

df_user_rec_movie = pd.merge(df_user_rec, df_movie, on="item")

# Print results.

for movie in df_user_rec_movie["title"]:
    print(movie)

print("It takes " + str(time.time() - time_start))