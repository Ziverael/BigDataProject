###MODULES###
import tensorflow as tf
from spark_tensorflow_distributor import MirroredStrategyRunner
from pyspark.ml.feature import StringIndexer
from PIL import Image, ImageOps
from time import time
from functools import wraps
import os
import io
import pickle
import numpy as np



###FUNCTIONS###
def log_timediff(time_, time2 = None):
    return f"Running time: {round(time() - time_, 3)} sec" if time2 is None else \
        f"Running time: {round(time() - time_, 3)} sec; Time from last check {round(time() - time2, 3)}"


def load_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return img, label


def dataframe_to_dataset(df, batch_size=32):
    df = df.copy()
    labels = df.pop('labels')
    filepaths = df.pop('filepaths')
    ds = tf.data.Dataset.from_tensor_slices((filepaths.values, labels.values))
    ds = ds.map(lambda x, y: load_image(x, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def file_exists(filename, base):
    return os.path.exists(os.path.join(base, filename))


def ordEncoding(df, inpCol, outCol):
    return StringIndexer(inputCol = inpCol, outputCol = outCol)\
        .fit(df)\
        .transform(df)


def train_decorator(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        def make_datasets():
            #TODO laod our dataset
            (mnist_images, mnist_labels), _ = \
                tf.keras.datasets.mnist.load_data(path=str(uuid.uuid4())+'mnist.npz')

            dataset = tf.data.Dataset.from_tensor_slices((
                tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
                tf.cast(mnist_labels, tf.int64))
            )
            dataset = dataset.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
            return dataset
        
        train_datasets = make_datasets()
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_datasets = train_datasets.with_options(options)

        return fun(*args, train_datasets=train_datasets, **kwargs)
    
    return wrapper


def load_preprocess_image_to_numpy(filepath):
    try:
        with open(filepath, 'rb') as f:
            img = Image.open(f)
            # Preprocessing steps
            img = ImageOps.grayscale(img)  # Convert to grayscale
            img = img.resize((128, 128))  # Resize image to 128x128 pixels
            img_array = np.array(img)  # Convert to numpy array
            img_array = img_array.astype(np.float32) / 255.0  # Normalize to [0, 1]
            return img_array.flatten().tolist()  # Flatten and convert to list for Spark
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return None


def save_chunks(df, chunk_size, path):
    total_rows = df.count()
    num_chunks = (total_rows // chunk_size) + 1
    for i in range(num_chunks):
        chunk_df = df.limit(chunk_size).offset(i * chunk_size)
        chunk_pd_df = chunk_df.toPandas()
        with open(f"{path}_chunk_{i}.pickle", "wb") as f:
            pickle.dump(chunk_pd_df, f)


def load_chunks(path):
    chunks = []
    i = 0
    while True:
        file_path = f"{path}_chunk_{i}.pickle"
        if not os.path.exists(file_path):
            break
        with open(file_path, "rb") as f:
            chunk = pickle.load(f)
            chunks.append(chunk)
        i += 1
    return pd.concat(chunks)