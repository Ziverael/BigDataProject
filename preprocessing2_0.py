from pyspark.sql import SparkSession
from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from functools import reduce
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from sparkdl import readImages as sparkdl_readImages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pyspark
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import os
from pyspark.sql.functions import lit
from pyspark.sql.functions import concat
import tensorflow as tf 
from spark_tensorflow_distributor import MirroredStrategyRunner
from pyspark.ml.feature import StringIndexer
from time import time
from functools import wraps
import pyspark.sql.functions as F
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
import pickle
from time import time
import logging
import pytz
from datetime import datetime
from sys import stdout


LOGS_FILE = r"/home/jacek/shared-drives/C:/Users/Jacek/projekt_big_data/Logs/training.log"

class LogFormatter(logging.Formatter):
    def format(self, record):
        warsaw_timezone = pytz.timezone('Europe/Warsaw')
        warsaw_time = datetime.now(warsaw_timezone)
        record.warsaw_time = warsaw_time.strftime('%Y-%m-%d %H:%M:%S')
        return super().format(record)
    
def save_chunks(df,label,path):
        with open(f"{path}_chunk_{label}.pickle", "wb") as f:
                pickle.dump(df, f)

def convert_bgr_array_to_rgb_array(img_array):
    B, G, R = img_array.T
    return np.array((R, G, B)).T

path = r'/home/jacek/shared-drives/C:/Users/Jacek/projekt_big_data/Data/Birds'

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



spark = SparkSession.builder.appName('Bird_Classification_test').config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.2") \
    .getOrCreate()
logger.info(f"spark is running")
logger.info("Loading dataset...")

bird_df = spark.read.csv(path=r'/home/jacek/shared-drives/C:/Users/Jacek/projekt_big_data/Data/Birds/birds.csv',
                        sep=',',
                        encoding='UTF-8',
                        comment=None,
                        header=True, 
                        inferSchema=True)
# bird_df.show(n=5, truncate=False)

# tutaj musicie zdefiniować swoje ścieżki, czyli gdzie trzymacie te zdjęcia 
photo_path = r"/home/jacek/shared-drives/C:/Users/Jacek/projekt_big_data/"
logger.debug("Loaded image hast table")

bird_df = bird_df.withColumn("filepaths", concat(lit(photo_path), bird_df["filepaths"]))

logger.debug(f"Dataset length {bird_df.count()}")
logger.info("Loaded dataset")

logger.info("Transforming data...")
logger.debug("Removing invalid paths")


distinct_labels = bird_df.select("labels").distinct().rdd.map(lambda row: row[0]).collect()
# index = distinct_labels.index("PARAKETT AKULET")
# distinct_labels = [distinct_labels[index]]

for label in distinct_labels:
    # train_df = bird_df.filter(bird_df["data set"] == "train").select('filepaths', 'labels').filter(bird_df["labels"] == label)
    # train_df = bird_df.filter(bird_df["data set"] == "test").select('filepaths', 'labels').filter(bird_df["labels"] == label) # This is test
    train_df = bird_df.filter(bird_df["data set"] == "valid").select('filepaths', 'labels').filter(bird_df["labels"] == label) # This is valid
    file_paths = [row.filepaths for row in train_df.collect()]
    logger.debug(f"Dataset length {train_df.count()}")


    img_df = spark.read.format("image").load(file_paths)

    def image_to_vec(image):
        # Convert to numpy array and flatten
        return DenseVector(ImageSchema.toNDArray(image).flatten())
    img2vec = udf(image_to_vec, VectorUDT())
    ImageSchema.imageFields
    # img_df

    df_with_vecs = img_df.withColumn('vecs', img2vec('image'))

    # logger.debug("Ordinal encoding...")
    rows = df_with_vecs.collect()
    logger.info("Transformed data")

  

    img_np_list= []
    for row in rows:
        img_dict = row['image']
        img_vec = row['vecs']

        # Extract image metadata
        width = img_dict['width']
        height = img_dict['height']
        nChannels = img_dict['nChannels']

        # Reshape vector to image array
        img_np = np.array(img_vec).reshape(height, width, nChannels)
        img_np_list.append(img_np)
        # Apply masking
        m = np.ma.masked_greater(img_np, 100)
        m_mask = m.mask
        args = np.argwhere(m_mask)

        #print(f"Processing image: {img_dict['origin']}")
        for idx, (r, c, _) in enumerate(args):
            # print(f"Row: {r}, Column: {c}, Value: {img_np[r, c]}")
            if idx > 5:  # Limit to first 5
                break



    final_train_df = [convert_bgr_array_to_rgb_array(i) for i in img_np_list]

    final_train_df = {label : final_train_df}
    logger.info(f"Pickling data: {label}")

    save_chunks(final_train_df,label,f'{path}/valid') #choose from train, test, valid
    # with open(r'/home/jacek/shared-drives/C:/Users/Jacek/projekt_big_data/Data/Birds/train.pickle', "wb") as f:
    #     pickle.dump(final_train_df, f)


logger.info("Preprocessing finished")
