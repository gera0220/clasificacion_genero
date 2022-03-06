from scipy.io.wavfile import read 
import numpy as np
from python_speech_features.base import mfcc
import matplotlib.pyplot as plt

fs, audio = read('prueba.wav')

print(f'Frecuencia de muestreo: {fs}')
print(f'Dimensiones del audio: {audio.shape[0]} puntos')

audio = audio[0:fs]

# Si graficamos el audio tenemos el siguiente diagrama.
plt.figure(1, figsize = (15, 5))
plt.plot(audio, color = "black")
plt.grid()
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

# Vamos a obtener sus coeficientes.
# Parámetros que recibe:
#   - signal: la señal de audio.
#   - winlen: tamaño de la ventana en segundos.
#   - winstep: tamaño del avance (se utiliza para el prcentaje de muestras que habrá entre ventanas). En este caso se tiene un overlap de 50%.
#   - numcep: número de coeficientes de Mel.
#   - nfilt: número de filtros del banco.
coeffs = mfcc(signal = audio, samplerate = fs, winlen = 0.030, winstep = 0.015, numcep = 25, nfilt = 30)

# Vamos a verlos en un gráfico
plt.figure(2, figsize=(15,4))
plt.imshow(coeffs.T, cmap="jet")
plt.show()

# A esta matriz hay que pegarle la clase a la que corresponde, como una columna más, esto es muy fácil ya que simplemente generamos un vector con el mísmo número (en este caso le estoy pegando puros ceros)

classes = np.matrix([0]*coeffs.shape[0])

# Lo pegamos a la matriz.
features = np.hstack([coeffs, classes.T])

print(features.shape) # este archivo ya puede guardarse en la computadora.
np.savetxt("mfcc_guitar.txt", features, delimiter=",", fmt="%f")
