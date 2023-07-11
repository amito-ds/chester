import os
import librosa
import torch
import numpy as np

# Load the SERNet model
model = torch.hub.load('pyannote/pyannote-audio', 'SER_0')

# Define a list of emotion labels
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Define the path to the directory containing the audio files
data_dir = "/Users/amitosi/PycharmProjects/chester/udi/emotion"

# Define empty lists to store the emotion labels and probabilities
emotion_labels = []
emotion_probs = []

# Set the model to evaluation mode
model.eval()

# Disable gradient calculation to speed up inference
torch.set_grad_enabled(False)

# Iterate through the audio files in the directory
for file in os.listdir(data_dir):
    try:
        # Load the audio data and resample it to 16 kHz (the same sample rate used during training)
        filepath = os.path.join(data_dir, file)
        audio_dat, sample_rate = librosa.load(filepath, sr=16000)
        audio_dat = librosa.resample(audio_dat, sample_rate, 16000)

        # Extract the features from the audio data using the SERNet model
        features = model({'audio': audio_dat})

        # Compute the probability distribution over the emotion classes
        probs = torch.nn.functional.softmax(features['log_likelihood']).numpy()[0]

        # Choose the emotion with the highest probability as the predicted emotion
        pred_emotion = emotions[np.argmax(probs)]

        # Append the predicted emotion label and probability to the respective lists
        emotion_labels.append(pred_emotion)
        emotion_probs.append(probs)
    except Exception as e:
        print(f"Error encountered while parsing {file}: {e}")
