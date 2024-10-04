import librosa
import numpy as np
import pandas as pd

from cough_segmentation.components.data_transformation import DataTransformation
from cough_segmentation.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from cough_segmentation.utils.utils import load_pickle_model
from cough_segmentation.entity.config_entity import ModelPredictionConfig
from cough_segmentation.logger import logging

class ModelPrediction:
    def __init__(self,model_prediction_config: ModelPredictionConfig, model_trainer_config: ModelTrainerConfig):
        self.model_prediction_config = model_prediction_config
        self.model_trainer_config = model_trainer_config
        
    def load_audio_file(self,file_path):    
        data = {"amp":[], "amp_original":[], "sf":[], "sf_original":[], "shape":[], "shape_original":[], "cough_start_end":np.nan, "label":np.nan}

        amp, sf = librosa.load(file_path)
        sf_resampled = 16000
        amp_resampled = librosa.resample(amp, orig_sr=sf, target_sr=sf_resampled)

        data["amp"].append(amp_resampled)
        data["amp_original"].append(amp)
        data["sf"].append(sf_resampled)
        data["sf_original"].append(sf)
        data["shape"].append(amp_resampled.shape)
        data["shape_original"].append(amp.shape)

        df = pd.DataFrame(data)
        return df


    def apply_framing(self,audio_df):
        def create_overlapping_frames(key, amp, label, sf, frame_size, hop_length):
            # Ensure amp is a numpy array
            amp = np.array(amp)

            total_frames = 1 + int((len(amp) - frame_size) / hop_length)
            dic = {"key":[], "sf":[], "start":[], "end":[], "max_amp":[], "frame_index":[], "amp":[], "label":np.nan}

            # Create overlapping frames
            for i in range(total_frames):
                dic["key"].append(key)
                dic["sf"].append(sf)
                dic["start"].append(i * hop_length)
                dic["end"].append(i * hop_length + frame_size)
                dic["max_amp"].append(np.max(amp[i * hop_length: i * hop_length + frame_size]))
                dic["frame_index"].append(i)
                dic["amp"].append(amp[i * hop_length: i * hop_length + frame_size])


            return dic

        def create_and_label_frames(row):
            return create_overlapping_frames(row.name, row["amp"], row["label"], row["sf"], frame_size, hop_length)

        all_frames = {}
        for frame_size in [1024]:
            hop_length = frame_size // 2

            frame_df = pd.DataFrame()
            xx = audio_df.apply(create_and_label_frames, axis=1)
            for x in xx:
                if len(frame_df) > 0:
                    frame_df = pd.concat([frame_df, pd.DataFrame(x)], ignore_index=True)
                else:
                    frame_df = pd.DataFrame(x)

            all_frames[frame_size] = frame_df
            print(f'Frame size {frame_size}, hop_length {hop_length}, count {len(frame_df)}')
        return all_frames

    def predict_audio(self,file_path):
        data_transform = DataTransformation(DataTransformationConfig)
        
        logging.info("loading the audio file")
        df = self.load_audio_file(file_path)
        logging.info("applying the framing")
        all_frames = self.apply_framing(audio_df=df)
        framed_df = all_frames[1024]
        final_df = framed_df.apply(data_transform.extract_features_combined, axis=1, result_type='expand')
        
        model = load_pickle_model(self.model_prediction_config.TRAINED_MODEL)
        transformer = load_pickle_model(self.model_trainer_config.TRANSFORMER_FILE_NAME)
        trf_data = transformer.transform(final_df)
        logging.info("predicting the data....")
        prediction = model.predict(trf_data)

        return prediction
