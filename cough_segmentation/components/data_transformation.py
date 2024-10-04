import os
import sys
import numpy as np
import pandas as pd
import librosa


from cough_segmentation.logger import logging
from cough_segmentation.exception import CoughSegmentationException
from cough_segmentation.entity.config_entity import DataTransformationConfig
from cough_segmentation.entity.artifact_entity import DataTransformationArtifacts, DataIngestionArtifacts

import pandas as pd
import numpy as np
import librosa
from joblib import Parallel, delayed

class DataTransformation:
    def __init__(self,data_transformation_config: DataTransformationConfig):
        self.data_transformation_config = data_transformation_config



    # Function to extract features
    def extract_features_combined(self,df):
        y = df['amp']
        sr = df['sf']
        features = {}

        # Time-domain features
        features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
        features['rms'] = np.mean(librosa.feature.rms(y=y))

        # Mel-frequency cepstral coefficients (MFCC)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=1024)
        
        for i in range(mfcc.shape[0]):
            features[f'mfcc_{i+1}'] = np.mean(mfcc[i])
        return features

    def preprocess_data(self):
        try:
            
            logging.info("Preprocessing the data ...")
            manual_data_workflow = pd.read_csv(self.data_transformation_config.SOURCE_MANUAL_DATA)
            files = manual_data_workflow["uuid"][manual_data_workflow["Training"] == 1].values
            audio_files = os.path.join(self.data_transformation_config.AUDIO_FILES)
            audio_df = self.get_audio_files_and_cough_timeframe_from_csv(manual_data_workflow, audio_files ,filename_index='uuid', start_end_time_prefix='IN_OUT_', max_start_end_count=16)
            audio_df = audio_df.loc[files]
            print(audio_df)
            audio_df.to_csv(self.data_transformation_config.PRE_TRANSFORMED_DATA_FILE_NAME,index=False)
            
            print("-----------------------------------------------------------------")
            logging.info(f"final audio shape is {audio_df.shape}")
            print("-----------------------------------------------------------------")
            # audio_df2 = audio_df.dropna()
            # audio_df2.reset_index(inplace=True)
            all_frames = self.apply_framing(audio_df=audio_df)
            framed_df = all_frames[self.data_transformation_config.AUDIO_FRAMES]
            logging.info("Saving the pre transformed data")
            logging.info(f"Post Transformation Stage Started with Feature Extraction Technique: {self.data_transformation_config.TECHNIQUE}")
            logging.info("extracting feature...")
            # framed_df[self.data_transformation_config.TECHNIQUE] = framed_df["amp"].apply(lambda x: self.extract_features(x))

            final_df = framed_df.apply(self.extract_features_combined, axis=1, result_type='expand')
            logging.info("feature extracted")
            result_df = pd.concat([framed_df, final_df], axis=1)
            result_df.to_csv(self.data_transformation_config.POST_TRANSFORMED_DATA_FILE_NAME,index=False)

            # audio_df_with_mel_spectrograms = self.compute_mel_spectrograms(framed_df)

            # data = audio_df_with_mel_spectrograms["mel"].values
            # mel_spectrograms = np.array([mel_spec.reshape(64, 16, 1) for mel_spec in data])
            
            # np.save(self.data_transformation_config.POST_TRANSFORMED_DATA_FILE_NAME,mel_spectrograms)
            # np.save(self.data_transformation_config.LABEL,audio_df_with_mel_spectrograms["label"].values)
            
            logging.info("Saving the post transformed data")

        except Exception as e:
            raise CoughSegmentationException(e,sys)
        
    def extract_features(self,amplitude):
        sr = 22050

        mfccs = librosa.feature.mfcc(y=amplitude, sr=sr, n_mfcc=13)
        mfccs_second_derivative = librosa.feature.delta(mfccs, order=2, mode="mirror",width=9)
        
        return mfccs_second_derivative
    
    def compute_mel_spectrograms(self,pd_df_audio_data):
        # Extract audio data and sample rates
        audio_frames = pd_df_audio_data["amp"].apply(lambda x: np.array(x) if not isinstance(x, np.ndarray) else x)
        sample_rates = pd_df_audio_data["sf"].values

        # Calculate mel-spectrograms
        mel_spectrograms = []
        mel_shapes = []
        for audio_frame, sr in zip(audio_frames, sample_rates):

            mel = librosa.feature.melspectrogram(y=audio_frame, sr=sr, n_fft=1024, win_length=1024, hop_length=68, n_mels=64)
            mel_db = librosa.power_to_db(mel, ref=np.max(mel))
            mel_spectrograms.append(mel_db)
            mel_shapes.append(mel_db.shape)

        # Create new dataframe for mel-spectrograms
        mel_spec_df = pd.DataFrame({
            "mel": mel_spectrograms,
            "mel_shape": mel_shapes,
            "label": pd_df_audio_data["label"].values
        }, index=pd_df_audio_data.index)

        # Concatenate additional columns from audio files dataframe
        columns_to_add = ["key", "sf", "start", "end", "max_amp", "frame_index", "amp", "label"]
        mel_spec_df = pd.concat([mel_spec_df, pd_df_audio_data[columns_to_add]], axis=1)
        
        return mel_spec_df

    def data_transformation(self):
        try:
            
            logging.info("Starting Data Transformation...")
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR,exist_ok=True)
            self.preprocess_data()
            logging.info(" Data Transformation Completed")

        except Exception as e:
            raise CoughSegmentationException(e,sys)
        
    # def label_cough_frames(self,time_frames, amp, sf):
    #     y = np.zeros(amp.shape[0])
    #     #print('len', amp.shape[0])
    #     for x in time_frames:
    #         start = int(x[0]*sf)
    #         end = int(x[1]*sf)
    #         y[start:end+1] = 1
    #         #print('start', f'{x[0]} sec ~ pos ', start, 'end', f'{x[1]} sec ~ pos ', end )
    #     return y

    # def get_audio_files_and_cough_timeframe_from_csv(self,df, audio_file_path, filename_index='uuid', start_end_time_prefix='IN_OUT_', max_start_end_count=16):
    #     """
    #     Get file names based on spreadsheet (Dataset workflow.csv) and start end time of cough

    #     Parameters
    #     df (DataFrame): Contains filenames and start end time values of cough sounds
    #     filenames (list): List of filenames
    #     filename_index (str): Column name of the df that holds values of file names
    #     start_end_time_prefix (str): Optional prefix text in the column names for start and end time of cough sounds
    #     max_start_end_count (int): Maximum number of column names for start and end time of cough sounds
    #     """
    #     # get column names for start and end time
    #     col_names = [f'{start_end_time_prefix}{x}' for x in range(1, max_start_end_count + 1)]

    #     extract_uuid = lambda file_path: os.path.splitext(os.path.basename(file_path))[0]

    #     amplitude_and_sampling_freq = lambda file_path: librosa.load(file_path)

    #     dic = {
    #         "amp": [], "amp_original": [], "sf": [], "sf_original": [],
    #         "shape": [], "shape_original": [], "cough_start_end": [], "label": []
    #     }
    #     files = []

    #     for file_name in os.listdir(audio_file_path):
    #         file_path = os.path.join(audio_file_path, file_name)
    #         uuid = extract_uuid(file_path)

    #         times = df[df[filename_index] == uuid][col_names].dropna(axis=1).values
    #         frame = []
    #         if len(times) > 0:
    #             frame.extend([list(map(float, x.strip().split(','))) for x in times[0] if len(x.strip().split(',')) == 2])

    #         files.append(uuid)
    #         amp, sf = amplitude_and_sampling_freq(file_path)

    #         # resample audio file to 16Khz based on Supervisor Request
    #         sf_resampled = 16000
    #         amp_resampled = librosa.resample(amp, orig_sr=sf, target_sr=sf_resampled)

    #         dic["amp"].append(amp_resampled)
    #         dic["amp_original"].append(amp)
    #         dic["sf"].append(sf_resampled)
    #         dic["sf_original"].append(sf)
    #         dic["shape"].append(amp_resampled.shape)
    #         dic["shape_original"].append(amp.shape)

    #         if len(frame) > 0:
    #             dic["cough_start_end"].append(frame)
    #             dic["label"].append(self.label_cough_frames(frame, amp_resampled, sf_resampled))
    #         else:
    #             dic["cough_start_end"].append(np.nan)
    #             dic["label"].append(self.label_cough_frames(frame, amp_resampled, sf_resampled))

    #     return pd.DataFrame(dic, index=files)


    def apply_framing(self,audio_df):
        def create_overlapping_frames(key, amp, label, sf, frame_size, hop_length):
            # Calculate the total number of frames
            if len(amp) == len(label):
                total_frames = 1 + int((len(amp) - frame_size) / hop_length)
                dic = {"key":[], "sf":[], "start":[], "end":[], "max_amp":[], "frame_index":[], "amp":[], "label":[]}

                # Create overlapping frames
                for i in range(total_frames):
                    dic["key"].append(key)
                    dic["sf"].append(sf)
                    dic["start"].append(i * hop_length)
                    dic["end"].append(i * hop_length + frame_size)
                    dic["max_amp"].append(np.max(amp[i * hop_length: i * hop_length + frame_size]))
                    dic["frame_index"].append(i)
                    dic["amp"].append(amp[i * hop_length: i * hop_length + frame_size])
                    frames_label_raw = label[i * hop_length: i * hop_length + frame_size]

                    f_label = 0
                    if np.sum(frames_label_raw==1) > (len(frames_label_raw) / 2):
                        f_label = 1
                    dic["label"].append(f_label)

                return dic
            else:
                print('Error: non matching amp and labels', key, len(amp), len(label))

        def create_and_label_frames(audio_df):
            return create_overlapping_frames(audio_df.name, audio_df["amp"], audio_df["label"],audio_df["sf"],frame_size, hop_length)

        all_frames = {}
        for frame_size in [256, 512, 1024]:
            hop_length = frame_size // 2

            frame_df = pd.DataFrame()
            #xx = audio_df.head(1).apply(create_and_label_frames, axis=1)
            xx = audio_df.apply(create_and_label_frames, axis=1)
            for x in xx:
                if len(frame_df) > 0:
                    frame_df = pd.concat([frame_df, pd.DataFrame(x)], ignore_index=True)
                else:
                    frame_df = pd.DataFrame(x)

            all_frames[frame_size] = frame_df
            print(f'Frame size {frame_size}, hop_length {hop_length}, count {len(frame_df)}')
        return all_frames

    def label_cough_frames(self,time_frames, amp, sf):
        y = np.zeros(amp.shape[0])
        #print('len', amp.shape[0])
        for x in time_frames:
            start = int(x[0]*sf)
            end = int(x[1]*sf)
            y[start:end+1] = 1
            #print('start', f'{x[0]} sec ~ pos ', start, 'end', f'{x[1]} sec ~ pos ', end )
        return y

    def get_audio_files_and_cough_timeframe_from_csv(self,df, audio_file_path, filename_index='uuid', start_end_time_prefix='IN_OUT_', max_start_end_count=16):
        """
        Get file names based on spreadsheet (Dataset workflow.csv) and start end time of cough

            Parameters
            df (DataFrame): Contains of filenames and start end time values of cough sounds
            filenames (list): List of filenames
            filename_index (str): Column name of the df that holds values of file names
            start_end_time_prefix (str): Optional prefix text in the column names for start and end time of cough sounds
            max_start_end_count (int): Maximum number of column names for start and end time of cough sounds
        """
        # get column names for start and end time
        col_names = [f'{start_end_time_prefix}{x}' for x in range(1,17)]

        extract_uuid = lambda file_path: os.path.splitext(os.path.basename(file_path))[0]

        amplitude_and_sampling_freq = lambda file_path: librosa.load(file_path)
        #amplitude_and_sampling_freq = lambda file_path: (np.zeros(1),0)

        dic = {"amp":[], "amp_original":[], "sf":[], "sf_original":[], "shape":[], "shape_original":[], "cough_start_end":[], "label":[]}
        files = []
        for file_name in os.listdir(audio_file_path):
            file_path = os.path.join(audio_file_path,file_name)
            uuid = extract_uuid(file_path)

            times = df[ df[filename_index] == uuid ][col_names].dropna(axis=1).values
            frame = []
            if len(times) > 0:
                frame.extend([list(map(float, x.strip().split(','))) for x in times[0] if len(x.strip().split(',')) == 2])


            files.append(uuid)
            amp, sf = amplitude_and_sampling_freq(file_path)

            #resample audio file to 16Khz based on Supervisor Request
            sf_resampled = 16000
            amp_resampled = librosa.resample(amp, orig_sr=sf, target_sr=sf_resampled)

            dic["amp"].append(amp_resampled)
            dic["amp_original"].append(amp)
            dic["sf"].append(sf_resampled)
            dic["sf_original"].append(sf)
            dic["shape"].append(amp_resampled.shape)
            dic["shape_original"].append(amp.shape)

            if len(frame) > 0:
            #print(uuid, times)
                dic["cough_start_end"].append(frame)
                dic["label"].append(self.label_cough_frames(frame, amp_resampled, sf_resampled))
            else:
                dic["cough_start_end"].append(np.nan)
                #30-May-24
                dic["label"].append(self.label_cough_frames(frame, amp_resampled, sf_resampled))


        return pd.DataFrame(dic, index=files)