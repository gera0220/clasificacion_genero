from scipy.io.wavfile import read
import numpy as np
from python_speech_features.base import mfcc
import os
import glob
import tqdm as tqdm

# Cambio de directorio
os.chdir('../data/raw')

# Guardar lista de folders
folder_files = os.listdir()

# Variables por conveniencia para el for
num_ceps = 25
full_features = np.matrix([1] * (num_ceps + 2))
k = 0

for folder in tqdm.tqdm(folder_files):
    os.chdir(folder)
    target = 1 if 'male' in folder else 0
    audio_files = [audio for audio in glob.glob('*.wav')]
    for file in tqdm.tqdm(audio_files):
        # Leer archivo del folder i
        fs, audio = read(file)
        audio = audio[0:fs]
        coeffs = mfcc(signal = audio, samplerate = fs, winlen = 0.030, winstep = 0.015, numcep = num_ceps, nfilt = 30, nfft = 1500)

        # Matriz de 1's (male) o 0's (female)
        classes = np.matrix([target]*coeffs.shape[0])

        # Índices para identificar audio
        index = np.matrix([k]*coeffs.shape[0])

        # Combinar 'coeffs' y 'classes'
        features = np.hstack([index.T, coeffs, classes.T])

        # Combinar features actuales con anteriores
        full_features = np.vstack([full_features, features])

        # Incrementar índice
        k += 1
    os.chdir('../')

# Guardar full_features excepto los 1's de la primera fila
np.savetxt('../features/full_features.txt', full_features[1:, ], delimiter = ',', fmt = '%f')

