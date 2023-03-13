
# Libraries
import tracemalloc
import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import datetime
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv



# Sampling frequency
freq = 44100

# Recording duration
duration = 5

#defining record function 

def record(n):
    recording = sd.rec(int(duration * freq),
				samplerate=freq, channels=2)
    sd.wait()
    write(f"record_{x}.wav", freq, recording)



# Load_model unpacks
def load_model():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(
        "facebook/wav2vec2-base-960h")
#     warnings.filterwarnings('ignore')
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
#     warnings.filterwarnings('ignore')
    return tokenizer, model


tokenizer, model = load_model()
x = str(input("number of input"))
print("speak : somthing like MOVE LEFT/RIGHT LEG/ARM UP/DOWN")

try:
    record(x)  # mp is microphone.py script and record is function to record the microphone input
    # for 5 seconds.
except AttributeError:
    pass


print("done")

# speech pre-processing
speech, fs = sf.read(f"record_{x}.wav")
if len(speech.shape) > 1:
    speech = speech[:, 0] + speech[:, 1]
if fs != 16000:
    speech = librosa.resample(speech, fs, 16000)

# word correction


dic = {'M': 'MOVE','R':'RIGHT','L':'LEFT','U':'UP','D':'DOWN','A':'ARM','LE':'LEG'}

def matching(str1,str2):
  n = len(str2)
  count = 0
  for i in str2:
    if (str1.find(i)) :
      count = count+1
      str1.replace(i,'')
  if count > n/2 :
    return True
  else:
    return False

s = ''




# inference

input_values = tokenizer(speech, return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
tracemalloc.start()
start = datetime.datetime.now()
transcription = tokenizer.decode(predicted_ids[0])
end = datetime.datetime.now()
time_cons = end - start
memory_consm = tracemalloc.get_traced_memory()
current, peak = memory_consm
tracemalloc.stop()


for i in transcription.split():
    x = '-'
    l = 'MRLUAUD LE'
    for j in i:
      if i.find(j) != -1:
        x = j
    if x == '-':
      s = s + '###'
      continue 
    try:
      y = dic[x]
    except KeyError :
      pass
      if i[0] == 'L':
        if (matching('LEFT',i)) :
          s = s+' LEFT '
        else:
          s = s+' LEG '
      else:
        s = s + ' ' +dic[i[0]] +' '

print(transcription)
f = open("test_op.txt", "a")
f.write("\n" + s + " " + str(peak/1024) +
        "Kb  inference time = " + str((time_cons.microseconds)) + "microseconds")
f.close()
print(time_cons.microseconds)
