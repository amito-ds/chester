import librosa
import numpy as np
import pandas as pd

from chester.run.full_run import run as chester_run
from chester.run.user_classes import ModelRun, Data
from udi.speech_to_text.speech_to_text_class import SpeechToText


def run(audio_data, sample_rates, n_mfcc=40, labels=None,
        is_pre_model=True,
        is_train_model=True, model_run: ModelRun = None,
        is_post_model=True,
        is_model_weaknesses=True,
        speech_to_text=False,
        plot=True
        ):
    udi_collector = {}
    chester_collector = {}
    features = []

    if labels is None:
        is_train_model = False
        labels = pd.Series([1] * len(audio_data))

    for audio, sr in zip(audio_data, sample_rates):
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            mfccs_mean = np.mean(mfccs.T, axis=0)  # calculate mean of MFCCs
            features.append(mfccs_mean)
        except Exception as e:
            print(f"Error encountered while parsing audio: {e}")

    # create pandas DataFrame
    columns = [f"feat_{i}" for i in range(n_mfcc)]
    df = pd.DataFrame(features, columns=columns)
    df["target"] = labels

    # TODO apply feature scaling if necessary
    if is_train_model:
        chester_collector.update(chester_run(data_spec=Data(df=df, target_column='target'),
                                             is_feature_stats=True,
                                             is_pre_model=is_pre_model,
                                             model_run=model_run, is_model_training=is_train_model,
                                             is_model_weaknesses=is_model_weaknesses,
                                             is_post_model=is_post_model,
                                             plot=plot))

    udi_collector["df"] = df
    udi_collector.update(chester_collector)

    if speech_to_text:
        transcriptions = SpeechToText(audio_data=audio_data, sample_rates=sample_rates).run()
        udi_collector["speech to text"] = transcriptions

    return udi_collector
