import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from NeuralNetwork import *


# Initial definitons
N = np.array([3,3,1]) # Layer Input: 3 - Hidden Layer 3 Neurons - Output Layer
NDatosEntrenamiento = 5000
NDatosPrueba = 200
ETA = 0.8
ALFA = 0.2
ErrorMinimo = 5e-3
EPOCA = 10

#Construction of the Neural Network
NLAYERS, W, INC, E, O, Y, X,DELTA = CONSTRUCCION(N)
#X = np.array([0.1, 0.1, 0.1])#only for test TODO: Quitar after test
#O, Y = PROPAGACION(W, X, E, O, Y, N, NLAYERS)
#DELTA = RETROPROPAGACION(W, ERR, O, DELTA, N, NLAYERS)
#W, INC =  AJUSTAW(W,INC,DELTA,X,O,ETA,ALFA,N,NLAYERS)
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
            #print(ERR)
            sum = 0
            for a in range(N[NLAYERS-1]):
                sum = sum + ERR
            Er[t] = math.sqrt(sum * sum)
            DELTA = RETROPROPAGACION(W, ERR, O, DELTA, N, NLAYERS)
            W, INC = AJUSTAW(W,INC,DELTA,X,O,ETA,ALFA,N,NLAYERS)
        ERROR[n] = np.mean(Er)
        if ERROR[n] < ErrorMinimo:
            training = False

#Test de Red Neuronal con datos de prueba TODO: NO mames hay que hacer esto
OutNNT = np.array(NDatosPrueba)

for t in range(NDatosEntrenamiento,NDatosEntrenamiento+NDatosPrueba):
    if t == 0:
        X[0]=0
        X[1]=0
        X[2] = U[t]
    elif t == 1:
        X[0]= 0
        X[1]= OutNNT[t-1]
        X[2] = U[t]
    else:
        X[0] = OutNNT[t-2]
        X[1] = OutNNT[t-1]
        X[2] = U[t]
    O, Y = PROPAGACION(W, X, E, O, Y, N, NLAYERS)
    OutNNT[t] = Y
#Normalizar datos de prueba TODO: Esto esta muy facil

# Normalizacion de datos, convertilos a un valor entre 0 y 1.


# Grafica algo al menos
n1 = NDatosEntrenamiento
n2 = NDatosPrueba+NDatosEntrenamiento
n3 = 1
n4 = 200
TT = range(1, n2+1)

plt.subplot(2, 1, 1)
plt.plot(T[n3:n4],OutNN[n3:n4], label='Neural network')
plt.plot(T[n3:n4], y[n3:n4], label='Output')
plt.plot(T[n3:n4],U[n3:n4], label='Input')
plt.title('Datos Entrenamiento')
plt.xlabel('Datos')
plt.axis([n3, n4, 0, 1])
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(ERROR,label='Error')
plt.title('Datos Entrenamiento')
plt.xlabel('Datos')
plt.grid(True)
plt.legend()

plt.show()

#print(E)
#print(str(neu_max)+" "+str(NLAYERS-1))
#print("inp="+str(neu_max)+"NLAYERS-1="+str(NLAYERS-1))

# print(W)
# print(INC)

# sys.exit()
# # # Neural Network construction
# for layer in range(1, NLAYERS):
#     for neuron in range(1, N[layer]+1):
#         for input in range(1,N[layer-1]+2):
#             #print(" Layer="+str(layer)+" Neuron="+str(neuron))
#             #      for input in range(1,N[layer]+BIAS):
#             m=m+1
#             W[layer-1][input-1][neuron-1]=np.random.rand()
#             print(str(m)+" - Layer="+str(layer)+" Neuron="+str(neuron)+" Input="+str(input)+"->"+str(W[layer-1][input-1][neuron-1]))
#
# # print(W[:][:][0])
# # print(W[:][:][1])
# # print(W[:][:][2])
# print("------------------------------")
# print(W)