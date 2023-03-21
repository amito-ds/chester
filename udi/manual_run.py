import os
import pandas as pd
import librosa
import numpy as np

from chester.run.full_run import run_madcat
from chester.run.user_classes import Data, ModelRun

data_dir = "/Users/amitosi/PycharmProjects/chester/udi/data"
labels = []
features = []

for subdir, dirs, files in os.walk(data_dir):
    for file in files:
        try:
            filepath = os.path.join(subdir, file)
            label = os.path.basename(subdir)
            labels.append(label)
            audio_data, sample_rate = librosa.load(filepath)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)  # calculate mean of MFCCs
            features.append(mfccs_mean)
            print(f"MFCC shape for {file}: {mfccs_mean.shape}")
        except Exception as e:
            print(f"Error encountered while parsing {file}: {e}")
            labels.pop()

# Create a pandas dataframe from the features and labels lists
df = pd.DataFrame(features, columns=['feature_' + str(i) for i in range(40)])
df['target'] = labels

print(df.head())  # prints first 5 rows of the dataframe

collector = run_madcat(data_spec=Data(df=df, target_column='target'), model_run=ModelRun(n_models=10),
                       feature_types=
                       {'numeric': list(df.columns)[:-1], 'boolean': [], 'text': [], 'categorical': [], 'time': [],
                        'id': []},
                       is_feature_stats=True, is_pre_model=True, is_model_weaknesses=False)
