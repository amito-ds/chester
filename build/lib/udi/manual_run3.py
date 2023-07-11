import IPython
import torch
import torchaudio
from torchaudio.utils import download_asset

from udi.run import run  # Replace 'your_module' with the module name where the 'run' function is defined

# Print torch and torchaudio versions
print(torch.__version__)
print(torchaudio.__version__)

# Set random seed
torch.random.manual_seed(0)

# Set device to use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download speech file
SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

# Display the audio file
IPython.display.Audio(SPEECH_FILE)

# Load the waveform and sample rate from the speech file
waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

# Resample the waveform if sample_rate is not equal to the model's sample rate (16kHz)
if sample_rate != 16000:
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

# Create lists with the waveform and sample rate as their single elements
audio_data = [waveform.squeeze().cpu().numpy()]
sample_rates = [16000]

# Call the 'run' function with the audio data and sample rates
udi_collector = run(audio_data=audio_data, sample_rates=sample_rates, labels=None, speech_to_text=True)

# Print the predicted transcription from the udi_collector dictionary
print("Predicted Transcription:", udi_collector["speech to text"][0])
