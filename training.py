###MODULES###
from models import (
    CustomModel,
    InceptionV3_custom,
    InceptionV3_custom_non_train,
    CNN_base,
    CNN_from_scratch,
#   CustomCNN,
#   VGG16  
)
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



###MAIN###
#Run timer
timer = time()

#Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = LogFormatter(
    f"train[UTC:%(asctime)s][WARSAW:%(warsaw_time)s][%(levelname)s] %(message)s",
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

#Setup spark
logger.debug("Starting spark session...")
spark = SparkSession \
       .builder \
       .appName("Big Data Project") \
       .getOrCreate()
logger.info(f"spark is running")


for name, m in {
    "CustomCNN" : CustomModel,
    "InceptionV3_custom" : InceptionV3_custom,
    "InceptionV3_custom_non_train" : InceptionV3_custom_non_train,
    "CNN_base" : CNN_base,
    "CNN_from_scratch" : CNN_from_scratch,
    # "CustomCNN" : CustomCNN,
    # "VGG16" : VGG16
    }.items():
    logger.info("Loading models...")
    model = m()
    # logger.debug(model)
    

    logger.info("Training model...")
    model.train_model()
    
    logger.info("Model trained...")

    logger.info("Evaluating model...")
    #Here evaluations and metrics
    #TODO

    logger.info("Saving model...")
    # model.set_model(model_to_train)
    # model.save_model(name)
    logger.info("Model saved")

logger.info("All trainings finished")

#Final models comparison
