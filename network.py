#Moreno Santiago José Miguel
#Fecha: 17 de febrero del 2024
#Actualización: Nuevo Optimizador 
#Se añadio 2 métodos nuevos y se modifico el update_mini_batch


import random 
import numpy as np 

class Network(object): 

    def __init__(self, sizes):
        
 

        self.num_layers = len(sizes) 
        self.sizes = sizes  
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def set_momentum_rate(self, momentum_rate):
        self.momentum_rate = momentum_rate  # Define el momentum rate
    
    def initialize_velocities(self):
        self.v_w = [np.zeros(w.shape) for w in self.weights]  # Inicializa la velocidad para los pesos
        self.v_b = [np.zeros(b.shape) for b in self.biases]  # Inicializa la velocidad para los sesgos

 

    def feedforward(self, a):

        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        if test_data: 
            test_data = list(test_data)
            n_test = len(test_data)

        training_data = list(training_data)  
        n = len(training_data)
        for j in range(epochs): 
            random.shuffle(training_data) 
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] 
            for mini_batch in mini_batches:    
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                          
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j)) 

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights] 
        for x, y in mini_batch: 
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Actualización de las velocidades con momentum
        self.v_w = [self.momentum_rate * v_w - (eta / len(mini_batch)) * nw  # Actualiza la velocidad de los pesos
                    for v_w, nw in zip(self.v_w, nabla_w)]
        self.v_b = [self.momentum_rate * v_b - (eta / len(mini_batch)) * nb  # Actualiza la velocidad de los sesgos
                    for v_b, nb in zip(self.v_b, nabla_b)]
        # Actualización de pesos y sesgos utilizando las velocidades de cambio
        self.weights = [w + v_w for w, v_w in zip(self.weights, self.v_w)]  # Actualiza los pesos con las velocidades
        self.biases = [b + v_b for b, v_b in zip(self.biases, self.v_b)]  # Actualiza los sesgos con las velocidades


    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights] 
        
        # feedforward
        activation = x 
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b 
            zs.append(z) 
            activation = sigmoid(z) 
            activations.append(activation) 
        
        # backward pass
        delta = self.CrossEntropy_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
     

        for l in range(2, self.num_layers): 
            z = zs[-l]  
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):

 
        
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def CrossEntropy_derivative(self, output_activations, y): #Se actualizo la función la derivada de costo con la derivada de cross entropy

        return ((output_activations - y) / (output_activations * (1 - output_activations))) #Por ende el return fue diferente

#### Miscellaneous functions
def sigmoid(z):

    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):

    return sigmoid(z)*(1-sigmoid(z))
