###MODULES###
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from utils import train_decorator
from spark_tensorflow_distributor import MirroredStrategyRunner
from functools import wraps #TEMP
import io
import os
import pickle
import numpy as np


###VARIABLES###
DATA_DIR = f"{os.path.dirname(__file__)}/Data/Birds"
string_io = io.StringIO()




def load_image(filepath, label):
    #TODO it will be removed. We use it until you implement Spark transformations in preprocessing.py
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [244, 244])
    image /= 255.0  # Normalize to [0,1]
    return image, label


# def dataframe_to_dataset(df, batch_size=32):
#     df = df.copy()
#     labels = df.pop('ordLabels').astype('int32')
#     filepaths = df.pop('filepaths')
#     return 
#     ds = tf.data.Dataset.from_tensor_slices((filepaths.values, labels.values))
#     ds = ds.map(lambda x, y: load_image(x, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     ds = ds.shuffle(buffer_size=len(df))
#     ds = ds.batch(batch_size)
#     ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#     return ds

def dataframe_to_numpy(df):
    df = df.copy()
    labels = df.pop('ordLabels').astype('int32').values
    filepaths = df.pop('filepaths').values

    X = np.array(pandas_df['preprocessed_image'].tolist())
    X = X.reshape(-1, 128, 128, 1)  # Reshape to (num_samples, height, width, channels)
    print(images)

    return images, labels

def create_batches(images, labels, batch_size=32):
    # Shuffle the data
    images, labels = shuffle(images, labels)

    # Calculate the number of batches
    num_batches = len(images) // batch_size
    if len(images) % batch_size != 0:
        num_batches += 1

    # Create batches
    image_batches = np.array_split(images, num_batches)
    label_batches = np.array_split(labels, num_batches)

    return image_batches, label_batches


# ###MODELS###
class CustomModel:
    BUFFER_SIZE = 100
    BATCH_SIZE = 32
    def encode_labels(self):
        with open(f"{DATA_DIR}/train.pickle", "rb") as f:
            data_train = pickle.load(f)
        self.label_encoder = encode_labels(data_train)

    def train_decorator(fun):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            def make_datasets():
                with open(f"{DATA_DIR}/train.pickle", "rb") as f:
                    data_train = pickle.load(f)
                with open(f"{DATA_DIR}/vali.pickle", "rb") as f:
                    data_vali = pickle.load(f)
                
                # train_datasets = dataframe_to_dataset(data_train, batch_size=CustomModel.BATCH_SIZE)
                X = np.array(pandas_df['preprocessed_image'].tolist())
                X = X.reshape(-1, 128, 128, 1)  # Reshape to (num_samples, height, width, channels)
                # valid_datasets = dataframe_to_dataset(data_vali)
                # valid_datasets = dataframe_to_dataset(data_vali, batch_size=CustomModel.BATCH_SIZE)
                
                # return train_datasets, valid_datasets
                return train_datasets#, valid_datasets
            
            # train_datasets, vali_dataset = make_datasets()
            train_datasets = make_datasets()
            # options = tf.data.Options()
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            # train_datasets = train_datasets.with_options(options)

            return fun(*args, train_datasets=train_datasets, **kwargs)
        
        return wrapper

    def __init__(self):
        self.model = None
        self.isCompiled = True
        self.data = None

    def __str__(self) -> str:
        self.model.summary(print_fn=lambda x: string_io.write(x + '\n'))
        return string_io.getvalue()

    def __repr__(self) -> str:
        return str(self)
    
    def get_model(self):
        if self.isCompiled:
            return self.model
        else:
            raise RuntimeError("Model is not compiled")
    
    def set_data(self, data):
        CustomModel.DATASET = data
        print("Y\n" * 10)
        print(CustomModel.DATASET)
    
    @train_decorator
    def train(self, train_datasets):
        def build_and_compile_cnn_model():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(244, 244, 1)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax'),
            ])
            model.compile(
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                metrics=['accuracy'],
            )
            return model

        multi_worker_model = build_and_compile_cnn_model()
        multi_worker_model.fit(x=train_datasets, epochs=3, steps_per_epoch=5)
    
    def train_model(self):
        MirroredStrategyRunner(
            num_slots = 1,
            use_gpu = False).run(self.train)














# class CustomModel:

#     @wrapper
#     def train(self, train_datasets):
#         def build_and_compile_cnn_model():
#             model = tf.keras.Sequential([
#                 tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
#                 tf.keras.layers.MaxPooling2D(),
#                 tf.keras.layers.Flatten(),
#                 tf.keras.layers.Dense(64, activation='relu'),
#                 tf.keras.layers.Dense(10, activation='softmax'),
#             ])
#             model.compile(
#                 loss=tf.keras.losses.sparse_categorical_crossentropy,
#                 optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#                 metrics=['accuracy'],
#             )
#             return model

#         multi_worker_model = build_and_compile_cnn_model()
#         multi_worker_model.fit(
#             x = train_datasets,
#             epochs = 3,
#             steps_per_epoch = 5
#             )

#     def train_model(self):
        # MirroredStrategyRunner(num_slots=2, use_gpu=False).run(self.train)



# class CustomModel:
    
#     # def compile(self):
#     #     self.isCompiled = True
#     #     self.model.compile(
#     #         loss = tf.keras.losses.sparse_categorical_crossentropy,
#     #         optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001),
#     #         metrics = ['accuracy']
#     #     )

#     @train_decorator
#     def _train(self):
#         def build_and_compile_model():
#             model = tf.keras.Sequential([
#                 tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
#                 tf.keras.layers.MaxPooling2D(),
#                 tf.keras.layers.Flatten(),
#                 tf.keras.layers.Dense(64, activation='relu'),
#                 tf.keras.layers.Dense(10, activation='softmax'),
#             ])
#             model.compile(
#                 loss=tf.keras.losses.sparse_categorical_crossentropy,
#                 optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#                 metrics=['accuracy'],
#             )
#             return model

#         multi_worker_model = build_and_compile_cnn_model()
#         multi_worker_model.fit(
#             x = self.data["train"],
#             epochs = 3,
#             steps_per_epoch = 5
#             )

#     def train_model(self):
#         if not self.isCompiled or self.data is None:
#             raise RuntimeError("Model in not compiled or data is not provided")
#         MirroredStrategyRunner(
#             num_slots = 2,
#             use_gpu = False
#             )\
#             .run(self._train)


    # def set_model(self, model):
    #     """Load trained model"""
    #     self.model = model
    
    # def evaluate(self, test_df, verbose = True):
    #     test_datasets = dataframe_to_dataset(test_df,
    #     batch_size = batch_size,
    #     training = False
    #     )
    #     test_loss, test_accuracy = self.model.evaluate(test_datasets)
    #     if verbose:
    #         print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
    #     return test_loss, test_accuracy

    # def save_model(self, name):
    #     self.model.save(f'{name}.h5')


# class VGG16(CustomModel):
#     def __init__(self, input_size = (244, 244, 1)):
#         super().__init__()
#         _input = Input(input_size)

#         conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)
#         conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
#         pool1  = MaxPooling2D((2, 2))(conv2)

#         conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
#         conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
#         pool2  = MaxPooling2D((2, 2))(conv4)

#         conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
#         conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
#         conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
#         pool3  = MaxPooling2D((2, 2))(conv7)

#         conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
#         conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
#         conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
#         pool4  = MaxPooling2D((2, 2))(conv10)

#         conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
#         conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
#         conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
#         pool5  = MaxPooling2D((2, 2))(conv13)

#         flat   = Flatten()(pool5)
#         dense1 = Dense(4096, activation="relu")(flat)
#         dense2 = Dense(4096, activation="relu")(dense1)
#         output = Dense(1000, activation="softmax")(dense2)

#         vgg16_model  = Model(inputs=_input, outputs=output)
#         self.model = vgg16_model
    

# class CustomCNN(CustomModel):
#     def __init__(self, input_size = (244, 244, 1)):
#         super().__init__()
#         mod = tf.keras.Sequential([
#             tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
#             tf.keras.layers.MaxPooling2D(),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(10, activation='softmax'),
#         ])
#         self.model = mod
    