###MODULES###
from utils import (
    file_exists,
    load_preprocess_image_to_numpy,
    ordEncoding,
    save_chunks,
)
from models import (
    CustomModel,
#   CustomCNN,
#   VGG16  
)
from functools import partial
from pyspark.sql.functions import col, udf
from pyspark.sql.types import BooleanType, ArrayType, FloatType
import os
import sys
import logging
import pyspark
from pyspark.sql import DataFrame, SparkSession
from typing import List
import pyspark.sql.types as T
import pyspark.sql.functions as F
from sys import stdout
from datetime import datetime
import pytz
from spark_tensorflow_distributor import MirroredStrategyRunner

from pyspark.sql import SparkSession
from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from pyspark.ml.feature import StringIndexer
from functools import reduce
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
# from sparkdl import DeepImageFeaturizer
# from sparkdl import readImages as sparkdl_readImages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql.functions import lit
from pyspark.sql.functions import concat
import tensorflow as tf
from  time import time
import pickle


###VARIABLES###
# DATA_DIR = "/home/zive_bewise/Documents/ML/Data/bird525species"
DATA_DIR = f"{os.path.dirname(__file__)}/Data/Birds"
LOGS_DIR = f"{os.path.dirname(__file__)}/Logs"
LOGS_FILE = f"{LOGS_DIR}/training.log"


###CLASSESS###
class LogFormatter(logging.Formatter):
    def format(self, record):
        warsaw_timezone = pytz.timezone('Europe/Warsaw')
        warsaw_time = datetime.now(warsaw_timezone)
        record.warsaw_time = warsaw_time.strftime('%Y-%m-%d %H:%M:%S')
        return super().format(record)






###FUNCTIONS###
udf_file_exists = udf(partial(file_exists, base = f"/"), BooleanType())
udf_load_image = udf(load_preprocess_image_to_numpy, ArrayType(FloatType()))


###MAIN###
#Run timer
timer = time()

#Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = LogFormatter(
    f"preprocessing[WARSAW:%(warsaw_time)s][%(levelname)s] %(message)s",
datefmt = '%Y-%m-%d %H:%M:%S'
)
stdout_handler = logging.StreamHandler(stdout)
f_handler = logging.FileHandler(LOGS_FILE)
stdout_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)
logger.addHandler(f_handler)        
logger.addHandler(stdout_handler)
logger.info("Script started...")


spark = SparkSession \
    .builder \
    .appName("Big Data Project") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.2") \
    .getOrCreate()
logger.info(f"spark is running")

logger.info("Loading dataset...")
bird_df = spark.read.csv(
    path = f'{DATA_DIR}/birds.csv',
    sep = ',',
    encoding = 'UTF-8',
    comment = None,
    header = True,
    inferSchema = True
)

bird_df.show(
    n = 5,
    truncate = False
)
logger.debug("Loaded image hast table")

bird_df = bird_df.withColumn(
    "filepaths",
    concat(
        lit(f'{DATA_DIR}/'),
        bird_df["filepaths"]
    )
)
bird_df.show(
    n = 5,
    truncate = False
)
logger.debug(f"Dataset length {bird_df.count()}")
logger.info("Loaded dataset")

logger.info("Transforming data...")
logger.debug("Removing invalid paths")
bird_df = bird_df\
    .withColumn("file_exists", udf_file_exists(col("filepaths")))\
    .filter(col("file_exists"))

bird_df.show(
    n = 5,
    truncate = False
)
logger.debug(f"Dataset length {bird_df.count()}")

#Data transformation and preprocessing in spark
bird_df = bird_df.repartition(100).cache()
bird_df = bird_df.withColumn("images", udf_load_image("filepaths"))
logger.info("Transformed data")
logger.debug("Ordinal encoding...")
bird_df = ordEncoding(bird_df, "labels", "ordLabels")

logger.info("Splitting data...")
train_df = bird_df.filter(bird_df["data set"] == "train").select('filepaths', 'labels', "ordLabels", "images")
test_df = bird_df.filter(bird_df["data set"] == "test").select('filepaths', 'labels', "ordLabels", "images")
vali_df = bird_df.filter(bird_df["data set"] == "valid").select('filepaths', 'labels', "ordLabels", "images")
logger.info("Data splitted")


# train_pd_df = train_df.toPandas()
# test_pd_df = test_df.toPandas()
# vali_pd_df = vali_df.toPandas()

#For test purpose
chunk_size = 5000  # Adjust chunk size as needed
save_chunks(train_df, chunk_size, f"{DATA_DIR}/train")
save_chunks(test_df, chunk_size, f"{DATA_DIR}/test")
save_chunks(vali_df, chunk_size, f"{DATA_DIR}/vali")
# train_pd_df = train_df.toPandas().loc[:300, :]
# test_pd_df = test_df.toPandas().loc[:300, :]
# vali_pd_df = vali_df.toPandas().loc[:300, :]

# print("Cols", train_df.columns)
# logger.info("Pickling data")
# with open(f"{DATA_DIR}/train.pickle", "wb") as f:
#     pickle.dump(train_pd_df, f)
# with open(f"{DATA_DIR}/test.pickle", "wb") as f:
#     pickle.dump(test_pd_df, f)
# with open(f"{DATA_DIR}/vali.pickle", "wb") as f:
#     pickle.dump(vali_pd_df, f)

logger.info("Preprocessing finished")
