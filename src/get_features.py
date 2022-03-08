from scipy.io.wavfile import read
import numpy as np
from python_speech_features.base import mfcc
import matplotlib.pyplot as plt
import os

# Cambio de directorio
os.chdir('../data/raw')

# Guardar lista de folders
folder_files = os.listdir()

# Variables por conveniencia para el for
num_ceps = 25
full_features = np.matrix([1] * (num_ceps + 1))

for folder in folder_files:
    os.chdir(folder)
    target = 1 if 'male' in folder else 0
    audio_files = os.listdir()
    for file in audio_files:
        # Leer archivo del folder i
        fs, audio = read(file)
        audio = audio[0:fs]
        coeffs = mfcc(signal = audio, samplerate = fs, winlen = 0.030, winstep = 0.015, numcep = num_ceps, nfilt = 30, nfft = 1500)

        # Matriz de 1's (male) o 0's (female)
        classes = np.matrix([target]*coeffs.shape[0])

        # Combinar 'coeffs' y 'classes'
        features = np.hstack([coeffs, classes.T])

        # Combinar features actuales con anteriores y redefinir fake_matrix
        full_features = np.vstack([full_features, features])
    os.chdir('../')

# Guardar full_features excepto los ceros de la primera fila
np.savetxt('../features/full_features.txt', full_features[1:, ], delimiter = ',', fmt = '%f')

