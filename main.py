import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from NeuralNetwork import *


# Initial definitons
N = np.array([3,6,1]) # Layer Input: 3 - Hidden Layer 3 Neurons - Output Layer
NDatosEntrenamiento = 5000
NDatosPrueba = 200
ETA = 0.8
ALFA = 0.2
ErrorMinimo = 5e-3
EPOCA = 20

#Construction of the Neural Network
NLAYERS, W, INC, E, O, Y, X,DELTA = CONSTRUCCION(N)

# Obtener Datos de entrenamiento y prueba para la funcion
# F(z) = z^2 / z^2 -0.2z^1 +0.6
cuenta = 1
r = 10 * np.random.rand()
T = np.zeros(NDatosEntrenamiento+NDatosPrueba)
U = np.zeros(NDatosEntrenamiento+NDatosPrueba)
y = np.zeros(NDatosEntrenamiento+NDatosPrueba)

for t in range(NDatosEntrenamiento+NDatosPrueba):
    T[t] = t
    U[t] = r
    cuenta = cuenta + 1
    if cuenta > 25:
        r = 10 * np.random.rand()
        cuenta = 1
    if t == 1:
        y[t] = U[t]
    if t == 2:
        y[t] = 0.2*y[t-1] + U[t]
    if t > 2:
        y[t] = 0.2 * y[t - 1] - 0.6*y[t-2] + U[t]

#Escalamiento
m = y.min()
M = y.max()
y = (y-m) / (M-m)
U = (U-m) / (M-m)

#Entrenamiento
Er = np.zeros(NDatosEntrenamiento)
OutNN = np.zeros(NDatosEntrenamiento)
ERROR = np.zeros(EPOCA)
training = True

for n in range(EPOCA):
    print(n+1)
    if training == True:
        for t in range(NDatosEntrenamiento):
            if t == 0:
                X[0]=0
                X[1]=0
                X[2] = U[t]
            elif t == 1:
                X[0]= 0
                X[1]= OutNN[t-1]
                X[2] = U[t]
            else:
                X[0] = OutNN[t-2]
                X[1] = OutNN[t-1]
                X[2] = U[t]
            O, Y = PROPAGACION(W, X, E, O, Y, N, NLAYERS)
            OutNN[t] = Y
            ERR = y[t] - OutNN[t]
            sum = 0
            for a in range(N[NLAYERS-1]):
                sum = sum + ERR
            Er[t] = math.sqrt(sum * sum)
            DELTA = RETROPROPAGACION(W, ERR, O, DELTA, N, NLAYERS)
            W, INC = AJUSTAW(W,INC,DELTA,X,O,ETA,ALFA,N,NLAYERS)
        ERROR[n] = np.mean(Er)
        if ERROR[n] < ErrorMinimo:
            training = False

print(W)

#Test de Red Neuronal con datos de prueba
OutNNT = np.zeros(NDatosPrueba)
T = np.zeros(NDatosPrueba)
Ut = np.zeros(NDatosPrueba)

for t in range(NDatosPrueba):
    T[t] = t
    Ut[t] = r
    cuenta = cuenta + 1
    if cuenta > 25:
        r = 10 * np.random.rand()
        cuenta = 1
    if t == 0:
        X[0] = 0
        X[1] = 0
        X[2] = Ut[t]
    elif t == 1:
        X[0] = 0
        X[1] = OutNNT[t-1]
        X[2] = Ut[t]
    else:
        X[0] = OutNNT[t-2]
        X[1] = OutNNT[t-1]
        X[2] = Ut[t]
    O, Y = PROPAGACION(W, X, E, O, Y, N, NLAYERS)
    OutNNT[t] = Y

# Normalizacion de datos, convertilos a un valor entre 0 y 1.
y = (M-m)*y + m
U = (M-m)*U + m
OutNN = (M-m)*OutNN + m
OutNNT = (M-m)*OutNNT + m

# Grafica
n1 = NDatosEntrenamiento
n2 = NDatosPrueba+NDatosEntrenamiento
n3 = 1
n4 = 200
TT = range(1, n2+1)

plt.subplot(3, 1, 1)
plt.plot(T[n3:n4],OutNN[n3:n4], label='Neural network')
plt.plot(T[n3:n4], y[n3:n4], label='Output')
plt.plot(T[n3:n4],U[n3:n4], label='Input')
plt.title('Datos Entrenamiento')
plt.xlabel('Datos')
plt.axis([n3, n4,-2, 10])
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(T,OutNNT,label='Data Test NN')
#plt.plot(T, y[n1:n2], label='Output')
plt.plot(T,Ut, label='Input')
plt.title('Datos Prueba')
plt.xlabel('Datos')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(ERROR,label='Error')
plt.yscale('symlog')
plt.title('Datos Entrenamiento')
plt.xlabel('Datos')
plt.grid(True)
plt.legend()

plt.show()
