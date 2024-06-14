###MODULES###
import tensorflow as tf
from spark_tensorflow_distributor import MirroredStrategyRunner
from pyspark.ml.feature import StringIndexer
from time import time
from functools import wraps
import os


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





def _train(
    model,
    train,
    valid,
    batch_size = 64
    ):
    train_datasets = dataframe_to_dataset(train_pd_df, batch_size = batch_size)
    valid_datasets = dataframe_to_dataset(valid_pd_df, batch_size = batch_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_datasets = train_datasets.with_options(options)
    valid_datasets = valid_datasets.with_options(options)

    # Fit the model
    model.fit(
        train_datasets,
        validation_data = valid_datasets,
        epochs = 10,
        steps_per_epoch = len(train_pd_df) // batch_size,
        validation_steps = len(valid_pd_df) // batch_size
    )


def distributed_training(
    model,
    train,
    valid,
    batch_size = 64,
    num_slot = 2,
    use_gpu = False
    ):
    strategy = MirroredStrategyRunner(
        num_slots = num_slot,
        use_gpu = use_gpu,
        ).run(_train, model = model,  train = train, valid = valid, batch_size = batch_size)

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
