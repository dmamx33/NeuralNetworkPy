import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from NeuralNetwork import *


# Initial definitons
N = np.array([3,3,1]) # Layer Input: 3 - Hidden Layer 3 Neurons - Output Layer 1
# NLAYERS = N.size
# neu_max = np.max(N)         #Numero máximo de neuronas por cada
# inp_max = np.max(N[1::])   #Maximo numero de entradas sin contar capa inicial
# nout_lay = N[-1]
# BIAS = 1

NDatosEntrenamiento = 5000
NDatosPrueba = 200
ETA = 0.8
ALFA = 0.2
ErrorMinimo = 5e-3
EPOCA = 50


NLAYERS, W, INC, E, O, Y, DELTA = CONSTRUCCION(N)
X = np.array([0.1, 0.1, 0.1])#only for test
O, Y = PROPAGACION(W, X, E, O, Y, N, NLAYERS)
print(W)
print("E=")
print(E)
print("O=")
print(O)
print("Y=")
print(Y)
sys.exit()
######################################################
# Obtener Datos de entreanmiento y prueba
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

#Entrenamiento TODO: Hacer lo mas perro

#Test de Red Neuronal con datos de prueba TODO: NO mames hay que hacer esto

#Normalizar datos de prueba TODO: Esto esta muy facil

# Normalizacion de datos, convertilos a un valor entre 0 y 1.
m = y.min()
M = y.max()
# y = (y-m) / (M-m)
# U = (U-m) / (M-m)

# Grafica algo al menos
n1 = NDatosEntrenamiento
n2 = NDatosPrueba+NDatosEntrenamiento
n3 = 1
n4 = 200
TT = range(1, n2+1)

plt.plot(T[n3:n4], y[n3:n4], label='Output')
plt.plot(T[n3:n4],U[n3:n4], label='Input')
plt.title('Datos Entrenamiento')
plt.xlabel('Tiempo')
plt.axis([n3, n4, -2, 10])
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