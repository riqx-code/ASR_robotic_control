# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write

# Sampling frequency
freq = 44100

# Recording duration
duration = 10

#defining record function 

def record():
    recording = sd.rec(int(duration * freq),
				samplerate=freq, channels=2)
    sd.wait()
    write("record.wav", freq, recording)
    
print("start")
record()
print("end")