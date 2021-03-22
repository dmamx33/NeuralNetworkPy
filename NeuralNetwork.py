import numpy as np
import matplotlib.pyplot as plt


# Initial definitons
N = np.array([3,3,1]) # Layer Input: 3 - Hidden Layer 3 Neurons - Output Layer 1
NLAYERS = N.size
neu_max = np.max(N)         #Numero máximo de neuronas por cada
inp_max = np.max(N[1::])   #Maximo numero de entradas sin contar capa inicial
NDatosEntrenamiento = 5000
NDatosPrueba = 200
ETA = 0.8
ALFA = 0.2
ErrorMinimo = 5e-3
EPOCA = 50
BIAS = 1


# Neural network construction (Weights and Auxiliaries Matrices)
W = np.zeros((NLAYERS-1, neu_max+BIAS, inp_max))
INC = np.zeros((NLAYERS-1, neu_max+BIAS, inp_max))

for layer in range(NLAYERS-1):
    for neuron in range(N[layer+1]):
        for input in range(N[layer]+BIAS):
            W[layer][input][neuron] = np.random.rand()