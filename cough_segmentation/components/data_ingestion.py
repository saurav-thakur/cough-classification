import os
import py7zr
import shutil
import sys

from cough_segmentation.logger import logging
from cough_segmentation.exception import CoughSegmentationException
from cough_segmentation.entity.config_entity import DataIngestionConfig
from cough_segmentation.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig) -> None:
        self.data_ingestion_config = data_ingestion_config


    def copy_file(self):
        try:
            if not os.path.exists(self.data_ingestion_config.SOURCE_DATA_DIR):
                logging.info(f"{self.data_ingestion_config.SOURCE_DATA_DIR} does not contains dataset!!")
        except Exception as e:
            raise CoughSegmentationException(e,sys)
            
        try:
            if os.path.exists(self.data_ingestion_config.SOURCE_DATA_DIR):
                os.makedirs(self.data_ingestion_config.DATA_INGESTION_DIR,exist_ok=True)
                shutil.copy(src=self.data_ingestion_config.SOURCE_DATA_DIR,dst=self.data_ingestion_config.DATA_INGESTION_DIR)
                shutil.copy(src=self.data_ingestion_config.SOURCE_MANUAL_DATA,dst=self.data_ingestion_config.DATA_INGESTION_DIR)
                logging.info(f"copied the required data from {self.data_ingestion_config.SOURCE_DATA_DIR} to {self.data_ingestion_config.DATA_INGESTION_DIR}")

            else:
                logging.info(f"the required data is not present in the {self.data_ingestion_config.DATA_INGESTION_DIR} folder")
        
        except Exception as e:
            raise CoughSegmentationException(e,sys)

    def unzip_file(self):

        try:
            unzip_path = self.data_ingestion_config.DATA_INGESTION_DIR
            zipped_audio_files = os.path.join(self.data_ingestion_config.DATA_INGESTION_DIR,self.data_ingestion_config.AUDIO_ZIPPED_FILE)
            os.makedirs(unzip_path,exist_ok=True)
            with py7zr.SevenZipFile(zipped_audio_files, mode="r") as z:
                z.extractall(path=self.data_ingestion_config.DATA_INGESTION_DIR)
            logging.info("data extracted from the zip folder")

        except Exception as e:
            raise CoughSegmentationException(e,sys)
        

    def start_data_ingestion(self):
        logging.info("Data Ingestion Started")

        try:
            self.copy_file()
            logging.info(f"copying the required data")

            logging.info(f"unzipping the data")
            self.unzip_file()

        except Exception as e:
            raise CoughSegmentationException(e,sys)

        


    