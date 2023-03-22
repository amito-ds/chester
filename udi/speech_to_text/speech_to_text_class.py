import re
import warnings

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


# Use the model and processor as before

def warning_filter(message):
    typed_storage_warning = "TypedStorage is deprecated"
    weights_warning = "Some weights of .* were not initialized from the model checkpoint"

    if re.search(typed_storage_warning, str(message)) or re.search(weights_warning, str(message)):
        return False
    return True


warnings.filterwarnings("ignore")


class SpeechToText:

    def __init__(self, audio_data, sample_rates, model_name="facebook/wav2vec2-base-960h"):
        self.audio_data = audio_data
        self.sample_rates = sample_rates
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):

        if self.model_name == "facebook/wav2vec2-base-960h":
            processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
        transcriptions = []

        for waveform, sample_rate in zip(self.audio_data, self.sample_rates):
            waveform_tensor = torch.tensor(waveform).to(self.device)
            input_values = processor(waveform_tensor, return_tensors="pt", padding=True,
                                     sampling_rate=sample_rate).input_values.to(self.device)

            with torch.inference_mode():
                logits = model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            transcriptions.append(transcription[0])

        return transcriptions
