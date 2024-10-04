import sys
import os

from cough_segmentation.logger import logging
from cough_segmentation.exception import CoughSegmentationException
from cough_segmentation.components.data_ingestion import DataIngestion
from cough_segmentation.components.data_transformation import DataTransformation
from cough_segmentation.components.model_trainer import ModelTrainer

from cough_segmentation.entity.config_entity import DataIngestionConfig, DataTransformationConfig,ModelTrainerConfig


class TrainingPipeline:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
    
    def start_ingesting_data(self):
        logging.info("Starting the Training Pipline's Data Ingestion")

        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion.start_data_ingestion()
        except Exception as e:
            raise CoughSegmentationException(e,sys)
        
    def start_data_transformation(self):
        logging.info("Starting the Training Pipline's Data Transformation")
        try:
            transform_data = DataTransformation(data_transformation_config=self.data_transformation_config)
            transform_data.data_transformation()
        except Exception as e:
            raise CoughSegmentationException(e,sys)
        
    def start_training_model(self):
        logging.info("Starting the Training Pipline's Model Training")

        try:
            train_model = ModelTrainer(model_trainer_config=self.model_trainer_config)
            train_model.train_model()
        except Exception as e:
            raise CoughSegmentationException(e,sys)
        
    def run_pipeline(self):
        logging.info("Running the pipelines")

        try:
            logging.info("Data Ingestion Pipeline Started")
            self.start_ingesting_data()
            logging.info("Data Ingestion Pipeline Completed")

            logging.info("Data Transformation Pipeline Started")
            self.start_data_transformation()
            logging.info("Data Transformation Pipeline Completed")
            
            logging.info("Model Training Pipeline Started")
            self.start_training_model()
            logging.info("Model Training Pipeline Completed")

        except Exception as e:
            raise CoughSegmentationException(e,sys)
        
