import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import os
import sys
import tensorflow as tf

from cough_segmentation.logger import logging
from cough_segmentation.exception import CoughSegmentationException
from cough_segmentation.entity.config_entity import ModelTrainerConfig
from cough_segmentation.entity.artifact_entity import ModelTrainerArtifacts
from cough_segmentation.ml.model import cough_detection_model
from cough_segmentation.utils.sono_cross_val import CrossValSplit
from cough_segmentation.utils.utils import save_pickle_model,load_pickle_model

class ModelTrainer:

    def __init__(self,model_trainer_config: ModelTrainerConfig):
            self.model_trainer_config = model_trainer_config

    
    def train_model(self):
        os.makedirs(self.model_trainer_config.MODEL_TRAINER_DIR,exist_ok=True)
        logging.info("loading dataset")

        try:
            df_from_save = pd.read_csv(self.model_trainer_config.PRE_TRANSFORMED_DATA_FILE_NAME)
            result_df = pd.read_csv(self.model_trainer_config.POST_TRANSFORMED_DATA_FILE_NAME)

            model,mean_acc_score,mean_precision_score,mean_recall_score,mean_f1_score,mean_auc_roc_score,confusion_m,transformer = cough_detection_model(df_from_save,result_df)

            metrics = {
            "mean_acc_score":[mean_acc_score],
            "mean_precision_score":[mean_precision_score],
            "mean_recall_score":[mean_recall_score],
            "mean_f1_score": [mean_f1_score],
            "mean_auc_roc_score": [mean_auc_roc_score],
            }

            curr_mean_metrics = pd.DataFrame(metrics)

            if os.path.exists(self.model_trainer_config.METRICS_FILE):
                prev_mean_metrics = pd.read_csv(self.model_trainer_config.METRICS_FILE)

                self.save_model(model,curr_mean_metrics,prev_mean_metrics,transformer)
            else:
                save_pickle_model(model,self.model_trainer_config.MODEL_NAME)
                logging.info("Saved a new model. No previous model found")
                curr_mean_metrics.to_csv(self.model_trainer_config.METRICS_FILE,index=False)
                logging.info("Saving the current metrics")

        except Exception as e:
            raise CoughSegmentationException(e,sys)

        
    def save_model(self,model,curr_mean_metrics,prev_model_metric,transformer):

        try:
            if curr_mean_metrics["mean_f1_score"].values > prev_model_metric["mean_f1_score"].values:
                save_pickle_model(model,self.model_trainer_config.MODEL_NAME)
                save_pickle_model(transformer,self.model_trainer_config.TRANSFORMER_FILE_NAME)
                logging.info("New Model Metrics is higher than previous one. New Model Saved")
                curr_mean_metrics.to_csv(self.model_trainer_config.METRICS_FILE,index=False)
                logging.info(f"Saving the current metrics as {prev_model_metric['mean_f1_score'].values} is {'equal' if prev_model_metric['mean_f1_score'].values == curr_mean_metrics['mean_f1_score'].values else 'higher'} it is better than the previous one")

            else:
                logging.info(f"Old model metrics {prev_model_metric['mean_f1_score'].values} is {'equal' if prev_model_metric['mean_f1_score'].values == curr_mean_metrics['mean_f1_score'].values else 'higher'} than previous one {curr_mean_metrics['mean_f1_score'].values} . Model not saved")

        except Exception as e:
            raise CoughSegmentationException(e,sys)

        

