import numpy as np
import matplotlib.pyplot as plt
import sys
import math

def CONSTRUCCION(N):

    NLAYERS = N.size
    neu_max = np.max(N)         #Numero máximo de neuronas por cada
    inp_max = np.max(N[1::])   #Maximo numero de entradas sin contar capa inicial
    nout_lay = N[-1]
    BIAS = 1

    W = np.zeros((NLAYERS - 1, neu_max + BIAS, inp_max))
    INC = np.zeros((NLAYERS - 1, neu_max + BIAS, inp_max))
    m = 1  # quitar
    for layer in range(NLAYERS - 1):
        for neuron in range(N[layer + 1]):
            for input in range(N[layer] + BIAS):
                W[layer][input][neuron] = m/10#np.random.rand()  # quitar
                m = m + 1  # quitar
                # print(str(m) + " - Layer=" + str(layer+1) + " Neuron=" + str(neuron+1) + " Input=" + str(input+1) + "->" + str(W[layer][input][neuron]))

    E = np.zeros((inp_max, NLAYERS - 1))
    O = np.zeros((inp_max, NLAYERS - 1))
    Y = np.transpose(np.zeros(nout_lay))
    DELTA = np.zeros((inp_max, NLAYERS - 1))

    return NLAYERS, W, INC, E, O, Y, DELTA


def PROPAGACION(W,X,E,O,Y,N,NLAYERS):

    for layer in range(NLAYERS - 1):
        for neuron in range(N[layer + 1]):
            SUM = 0
            for input in range(N[layer] + 1):
                if layer == 0:
                    if input == N[layer]:
                        SUM = SUM + W[layer][input][neuron]
                    else:
                        SUM = SUM + W[layer][input][neuron] * X[input]
                else:
                    if input == N[layer]:
                        SUM = SUM + W[layer][input][neuron]
                    else:
                        SUM = SUM + W[layer][input][neuron] * O[input, layer - 1]
            E[neuron][layer] = SUM
            O[neuron][layer] = 1 / (1 + math.exp(-1 * E[neuron][layer]))

    for output in range(Y.size):
        Y[output] = O[output][NLAYERS - 2]

    return O, Y

def AJUSTAW(W,INC,DELTA,X,O,ETA,ALFA,N,NLAYERS):

    for layer in range(NLAYERS - 1):
        for neuron in range(N[layer + 1]):
            SUM = 0
            for input in range(N[layer] + 1):
                if layer == 0:
                    if input == N[layer]:
                        AUX = 1
                    else:
                        AUX = X[input]
                else:
                    if input == N[layer]:
                        AUX = 1
                    else:
                        AUX = O[neuron][layer-1]
                INC[layer][input][neuron] = ETA * DELTA[neuron][layer] * AUX + ALFA * INC[layer][input][neuron]
                W[layer][input][neuron] = W[layer][input][neuron] + INC[layer][input][neuron]