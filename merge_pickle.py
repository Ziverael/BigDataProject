import pickle
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.functions import concat

# List of pickle files to merge
spark = SparkSession.builder.appName('Bird_Classification').getOrCreate()
bird_df = spark.read.csv(path=r'/home/jacek/shared-drives/C:/Users/Jacek/projekt_big_data/Data/Birds/birds.csv',
                        sep=',',
                        encoding='UTF-8',
                        comment=None,
                        header=True, 
                        inferSchema=True)

photo_path = r"/home/jacek/shared-drives/C:/Users/Jacek/projekt_big_data/"
bird_df = bird_df.withColumn("filepaths", concat(lit(photo_path), bird_df["filepaths"]))
distinct_labels = bird_df.select("labels").distinct().rdd.map(lambda row: row[0]).collect()
path = r'/home/jacek/shared-drives/C:/Users/Jacek/projekt_big_data/Data/Birds/valid'#choose from train, test, valid
pickle_files = [f"{path}_chunk_{label}.pickle" for label in distinct_labels]

combined_data = {}

# Load data from each pickle file and combine it
for file in pickle_files:
    with open(file, 'rb') as f:
        data = pickle.load(f)
        combined_data.update(data)  # Assuming data is a dictionary

# Save the combined data to a new pickle file
with open(f'{path}_combined_file.pickle', 'wb') as f:
    pickle.dump(combined_data, f)
