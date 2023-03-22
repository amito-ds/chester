import os
import pandas as pd
import librosa
import numpy as np

from chester.run.full_run import run
from chester.run.user_classes import Data, ModelRun
from udi.run import run

# data_dir = "/Users/amitosi/PycharmProjects/chester/udi/data"
data_dir = "/Users/amitosi/PycharmProjects/chester/udi/data_yes_no"
labels = []
audio_data = []
sample_rates = []

for subdir, dirs, files in os.walk(data_dir):
    for file in files:
        try:
            filepath = os.path.join(subdir, file)
            label = os.path.basename(subdir)
            labels.append(label)
            audio_dat, sample_rate = librosa.load(filepath)
            audio_data.append(audio_dat)
            sample_rates.append(sample_rate)
        except Exception as e:
            print(f"Error encountered while parsing {file}: {e}")
            labels.pop()

# udi_collector = run(audio_data=audio_data, sample_rates=sample_rates, labels=labels, n_mfcc=100)
