import random

import librosa
import numpy as np

from chester.run.user_classes import FeatureStats, TextFeatureExtraction, ModelRun
from diamond.data_augmentation.augmentation import ImageAugmentation
from diamond.face_detection.face_image_ import ImageFaceDetection
from diamond.image_caption.image_caption_class import ImageDescription
from diamond.image_data_info.image_info import ImageInfo
from diamond.image_object_detection.image_object_class import ImageObjectDetection
from diamond.model_training.best_model import ImageModelsTraining
from diamond.post_model_analysis.post_model import ImagePostModelAnalysis
from diamond.user_classes import ImagesData, ImagesAugmentationInfo, ImageModels, ImagePostModelSpec, \
    ImageDescriptionSpec
import pandas as pd

from diamond.utils import index_labels


def run(audio_files, sample_rates, labels=None,
        feature_stats: FeatureStats = None,
        is_feature_stats=True, is_pre_model=True,
        is_model_training=True, model_run: ModelRun = None):
    udi_collector = {}

    if labels is None:
        is_train_model = False
        labels = pd.Series([1] * len(audio_files))

    # create the wanted pandas df using the audio files and labels
    df = pd.DataFrame({
        'label': labels
    })

    for i, file in enumerate(audio_files):
        try:
            # load the audio file
            audio_data, sample_rate = librosa.load(file, sr=sample_rates[i])

            # extract the features using MFCC
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)  # calculate mean of MFCCs

            # add the features to the dataframe
            feature_cols = ['feat_' + str(j) for j in range(40)]
            features_dict = dict(zip(feature_cols, mfccs_mean))
            features_dict['label'] = labels[i]
            df = df.append(features_dict, ignore_index=True)

            print(f"MFCC shape for {file}: {mfccs_mean.shape}")
        except Exception as e:
            print(f"Error encountered while parsing {file}: {e}")

    # apply feature scaling if necessary
    # if is_feature_stats and feature_stats is not None:
    #     df = scale_features(df, feature_stats)

    # train and run the model if necessary
    # if is_model_training and model_run is not None:
    #     model = train_model(df, model_run)
    #     udi_collector = model.predict(df.drop('label', axis=1))

    return udi_collector



    # create the wanted pandas df

    # load

    # pp

    # Extract features

    # Madcat

    # more?
    return udi_collector
