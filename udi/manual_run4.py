import os

import torch
import torchaudio

from udi.run import run  # Replace 'your_module' with the module name where the 'run' function is defined

# Print torch and torchaudio versions
print(torch.__version__)
print(torchaudio.__version__)

# Set random seed
torch.random.manual_seed(0)

# Set device to use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_path = "/Users/amitosi/PycharmProjects/chester/udi/emotion"

audio_data = []
sample_rates = []

for file in os.listdir(emotion_path):
    if file.endswith(".wav"):
        filepath = os.path.join(emotion_path, file)
        waveform, sample_rate = torchaudio.load(filepath)
        waveform = waveform.to(device)

        # Resample the waveform if sample_rate is not equal to the model's sample rate (16kHz)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        audio_data.append(waveform.squeeze().cpu().numpy())
        sample_rates.append(16000)

# Call the 'run' function with the audio data and sample rates
udi_collector = run(audio_data=audio_data, sample_rates=sample_rates, labels=None, speech_to_text=True)

# Print the predicted transcriptions from the udi_collector dictionary
for i, transcription in enumerate(udi_collector["speech to text"]):
    print(f"Predicted Transcription for file {i + 1}: {transcription}")
