import numpy as np

# Initial definitons
N = np.array([3, 3, 4, 1])                     # Layer Input: 3 - Hidden Layer 3 Neurons - Output Layer 1
NLAYERS = N.size
neu_max = np.max(N)
print(neu_max)
NDatosEntrenamiento = 5000
NDatosPrueba = 200
ETA = 0.8
ALFA = 0.2
ErrorMinimo = 3e-3
EPOCA = 100
BIAS = 1
m=0
W = np.zeros((neu_max+1,neu_max,NLAYERS-1))
print("Tamaño ="+str(W.size))
print(W)
# # Neural Network construction
for layer in range(1, NLAYERS):
    for neuron in range(1, N[layer]+1):
        for input in range(1,N[layer-1]+2):
            #print(" Layer="+str(layer)+" Neuron="+str(neuron))
            #      for input in range(1,N[layer]+BIAS):
            m=m+1
            W[input-1][neuron-1][layer-1]=m#np.random.rand()
            print(str(m)+" - Layer="+str(layer)+" Neuron="+str(neuron)+" Input="+str(input)+"->"+str(W[input-1][neuron-1][layer-1]))

print(W[:][:][0])
print(W[:][:][1])
print(W[:][:][2])
print("-----------------")
print(W)




