from scipy.io.wavfile import read
import numpy as np
from python_speech_features.base import mfcc
import matplotlib.pyplot as plt
import os

# Cambio de directorio
os.chdir('../data/raw')

# Guardar lista de folders
folder_files = os.listdir()

for folder in folder_files:
    os.chdir(folder)
    target = 1 if 'male' in folder else 0
    audio_files = os.listdir()
    for file in audio_files:
        # Leer archivo del folder i
        fs, audio = read(file)
        audio = audio[0:fs]

        # Vamos a obtener sus coeficientes.
        # Parámetros que recibe:
        #   - signal: la señal de audio.
        #   - winlen: tamaño de la ventana en segundos.
        #   - winstep: tamaño del avance (se utiliza para el prcentaje de muestras que habrá entre ventanas). En este caso se tiene un overlap de 50%.
        #   - numcep: número de coeficientes de Mel.        coeffs.shape
        #   - nfilt: número de filtros del banco.
        coeffs = mfcc(signal = audio, samplerate = fs, winlen = 0.030, winstep = 0.015, numcep = 25, nfilt = 30, nfft = 1500)

        # Matriz de 1's (male) o 0's (female)
        classes = np.matrix([target]*coeffs.shape[0])

        # Combinar 'coeffs' y 'classes'
        features = np.hstack([coeffs, classes.T])

        np.savetxt('../../features/mfcc_' + file.split('.')[0] + '.txt', features, delimiter=",", fmt="%f")
    os.chdir('../')
