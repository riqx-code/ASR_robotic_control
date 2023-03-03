# ***Hardware Modelling AZAD 2023***

## ***Automatic Speech Recognition*** for Robotic Control of Exoskeleton 

**Language** : Python >= 3.8.0

### List of Scripts 

- ***Microphone***
    - **Language** : Python
    - **Size**     : ###
    - **Function** : To take input of microphone for defined time and write it to a specific file(overwritten each time) of .wav format.
    - **Dependances** :
        - *sounddevice* : for recording at specific Fs and  Duration
        - *scipy.io.wavfile* : I used write function to write recorded audio in .wav format into a file.

    - **Location of File**
        ```
        ASR_control/microphone.py
        ```
- ***Inference***
    - **Language** : Python
    - **Size** : ###
    - **Function** : To take the recorded .wav file by microphone.py Script and do inference using facebooks-wav2vec2-model.

        And to store output in a text file in format of infered_text , memory usage during inference in Kb & time of inferece in microseconds.
    - **Dependances** :
        - *librosa* : To load the input audio file and preprocess the audio data.
        - *torch* : to work with Deep Learning architecture model 
        - [*transformers*](https://github.com/huggingface/transformers) : set of trained models with thier affiliate functions 
        - *tracemalloc* : to measure the memory required for inference 
        - *datetime* : to measure time for inference 
