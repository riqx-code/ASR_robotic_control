# Libraries

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import datetime
import tracemalloc



# Load_model unpacks
def load_model():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(
        "facebook/wav2vec2-base-960h")

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    return tokenizer, model

# unloading model
tokenizer, model = load_model()


# input for file : file number

x = str(input("file number"))


# speech pre-processing
speech, fs = librosa.load(f"record_{x}.wav") # loading the particular file

if len(speech.shape) > 1:
    speech = speech[:, 0] + speech[:, 1] # combining speech from different channels(sources)
if fs != 16000:
    speech = librosa.resample(speech, fs, 16000) # resampling to 16000

# inference
input_values = tokenizer(speech, return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
# tracemalloc.start()
# start = datetime.datetime.now()
transcription = tokenizer.decode(predicted_ids[0])
# end = datetime.datetime.now()
# time_cons = end - start
# memory_consm = tracemalloc.get_traced_memory()
# current, peak = memory_consm
# tracemalloc.stop()
# f = open("outputs_texts.txt", "a")
# f.write("\n" +f" record.wav :  " +transcription + " " + str(peak/1024) +
#         "Kb  inference time = " + str((time_cons.microseconds)) + "microseconds")
# f.close()
# print(time_cons.microseconds)