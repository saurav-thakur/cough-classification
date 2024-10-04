import os
from datetime import datetime


# common constants
TIMESTAMP: str =  datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
# to create the version of the folder.
# ARTIFACTS_DIR = os.path.join("artifacts",TIMESTAMP)
ARTIFACTS_DIR = "artifacts"
# the bucket name where the data has been stored in the cloud
BUCKET_NAME = "dataset.zip"

# original data
ORIGINAL_DATASET_DIR = "dataset"

# data ingestion
DATA_INGESTION_DIR = "data_ingestion"
SOURCE_DATA = "audio_files.7z"
SOURCE_MANUAL_DATA = "Dataset workflow.csv"

# data transformation
DATA_TRANSFORMATION_ARTIFACTS_DIR = "data_transformation"
AUDIO_FILE_PATH = "AWS Audio Files"
PRE_TRANSFORMED_DATA_FILE_NAME = "pre_transformed_data.feather"
POST_TRANSFORMED_DATA_FILE_NAME = "post_transformed_data.csv"
LABEL = "label.npy"
AUDIO_FRAMES = 1024
TECHNIQUE = "Combined Feature Extraction RMS MFCC and ZCR"


# model trainer constants
MODEL_TRAINER_DIR = "trained_model"
PRE_TRANSFORMED_DATA_FILE_NAME = "pre_transformed_data.csv"
POST_TRANSFORMED_DATA_FILE_NAME = "post_transformed_data.csv"
EPOCH = 10
FINAL_ACTIVATION = "sigmoid"
K_FOLD = 5
MODEL_NAME = "model.pickle"
METRICS_FILE = "metrics.csv"
TRANSFORMER_FILE_NAME = "transformer.pickle"

# prediction pipeline
MODEL_TRAINER_DIR = "trained_model"
MODEL_NAME = "model.pickle"
TESTING_DATA = "testing_data"