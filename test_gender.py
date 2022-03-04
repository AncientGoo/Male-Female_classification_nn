from importlib.resources import path
import tensorflow as tf
import numpy as np
import librosa
import sys



def predict_wav():

    modelpath = 'model'
    path_to_file = sys.argv[1]
    
    sound, _ = librosa.load(path_to_file)
        
    stft = librosa.stft(sound, n_fft = 2048, hop_length = 1024)
    stft_sum = np.expand_dims(np.sum(stft, axis=1), axis=1)
    stft_sum = np.abs(stft_sum)
    stft_sum = stft_sum / stft_sum.mean()
    
    audio = np.expand_dims(stft_sum, axis=0)
    
    model = tf.keras.models.load_model(modelpath)
    prediction = model.predict(audio)
    
    return prediction

predict_wav()