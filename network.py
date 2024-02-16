#Moreno Santiago José Miguel
#Fecha: 26 de enero del 2024
#Actualización: Documentación del código


import random #Importamos la librería para generar números pseudoaleatorios
import numpy as np #Importamos la librería de numpy para usar matrices, vectores y demás funciones matemáticas de forma eficiente.

class Network(object): #Definiremos la clase Network (las instrucciones para nuestra red neuronal) que heredará de objecto (el objeto es como una variable sofisticada)

    def __init__(self, sizes):
        
        """
        La función init se ejecutará cuando se cree un objeto de la clase Network con los siguientes parametros
        self: objeto en si mismo, cuando una función lleva self, estamos indicando que esa función pertecene a esa clase
        sizes: Una lista que contendrá el número de neuronas cada una de las capas de la red neuronal

        En otras palabras creamos un "constructor" de clase de si mismo para sizes
        """

        self.num_layers = len(sizes) #obtenemos la longitud de la lista para determinar el número de capas en la red neuronal
        self.sizes = sizes  #Almacenamos la lista de tamaños de capa como un atributo de si mismo
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        #Empezamos sizes de 1 ya que la capa de entrada no tiene bias ni pesos 
        #Las bias y los pesos en Network se inicializan usando la función np.random.randn de la librería Numpy
                                                                 
        
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        """
        Dado una cierto sizes = [x_1,x_2..., x_n] obtendremos su longitud de dicha lista para saber cuantas capas tendrá la red neuronal (sin contar la capa de entrada)
        Una vez almacenada dicha lista le asignamos un valor de peso y bias a cada neurona con numpy
        """    

    def feedforward(self, a):
        """
        Dada una entrada se procesa a través de cada capa usando las listas iteradas de pesos, bias y funciones de activación, y devuelve la salida de la red.
        """

        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        
        """
        Aplicamos un método de aprendizaje: descenso de gradiente estocástico

        Dado cada uno de los epoch, comienza mezclando aleatoriamente los datos de entrenamiento, y luego los divide en mini_batch del tamaño adecuado
        Luego para cada mini_batch aplicamos un solo paso de descenso de gradiente. 
        Usando self.update_mini_batch(mini_batch, eta)
        actualizamos los pesos y bias de la red ,según una sola iteración de descenso de gradiente, utilizando solo los datos de entrenamiento en el mini_batch
        
        Definimos a training_data como una lista (x, y) 
        donde
        x: datos de entrenamiento 
        y: las salidas deseadas correspondientes 
        Epochs es el número de épocas que se va a entrenar y mini_batch__size es el tamaño de estos para el entrenamietno
        eta es la tasa de aprendizaje, test_data: para evaluar la red después de cada epoch

        """

        if test_data: #Convierte en una lista los datos de prueba (test_data) si es que hay y obtiene la longitud de estos
            test_data = list(test_data)
            n_test = len(test_data)

        training_data = list(training_data) #Convierte en una lista los datos de entrenamiento si es que hay y oobtiene la longitud de estos
        n = len(training_data)
        for j in range(epochs): #Itera sobre el número de épocas especificadas
            random.shuffle(training_data) # Mezcla aleatoriamente los datos de entrenamiento en cada época.
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] # Divide los datos de entrenamiento
            for mini_batch in mini_batches:    # Itera sobre cada mini_batch y actualiza los pesos y sesgos
                self.update_mini_batch(mini_batch, eta)
            if test_data: # Si hay datos de prueba, usa print para mostrar la precisión de la red en los datos de prueba
                          
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j)) # Si no hay datos de prueba, muestra que la época ha sido completada

    def update_mini_batch(self, mini_batch, eta):

        """
         update_mini_batch calcula gradientes para cada ejemplo de entrenamiento en el mini_batch y despues actualiza self.weights y self.biases.
        """
        
        nabla_b = [np.zeros(b.shape) for b in self.biases] #Inicializamos una lista de matrizes en 0´s para los pesos
        nabla_w = [np.zeros(w.shape) for w in self.weights] #Inicializamos una lista de matrizes en 0´s para las bias
        for x, y in mini_batch: #Iteramos para cada elemento de la lista mini_batch
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        """
        Usamos la retropropagación (backprop) para calcular los gradientes de la función de costo con respecto a los parámetros de la red neuronal, 
        por ende podemos actualizar dichos parámetros durante el entrenamiento.
        """
        
        nabla_b = [np.zeros(b.shape) for b in self.biases] #Volvemos a iniciar los nablas en 0 ya que salen con valor distinto a 0
        nabla_w = [np.zeros(w.shape) for w in self.weights] #y los estamos iterando así que es conveniente reinciar esto
        
        # feedforward
        activation = x #Datos de entrada
        activations = [x] #Lista para almacenar todas las activaciones, capa por capa.
        zs = [] #Lista para almacenar todos los vectores z, capa por capa.
        for b, w in zip(self.biases, self.weights): # se está iterando simultáneamente sobre todas las matrices de pesos w y  los vectores de bias b de cada capa de la red neuronal.
            z = np.dot(w, activation)+b #Suma todos los w y x (pesos y datos de entrada) y a la sumatoria le añade la bias, es decir tenemos un valor de z para cada neurona
            zs.append(z) #Añadimos el valor de z a la lista z,s para obtener una lista que contenga todos los z de las capas
            activation = sigmoid(z) #La salida de la actual neurona va ser la entrada de las neuronas de la siguiente capa
            activations.append(activation) #Añadimos dicha salida en nuestra lista
        
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Nota que la variable l en el bucle debajo se utiliza de manera un poco
        # diferente a la notación en el Capítulo 2 del libro. Aquí,
        # l = 1 significa la última capa de neuronas, l = 2 es la
        # segunda última capa, y así sucesivamente. Es un renumeración del
        # esquema en el libro, utilizada aquí para aprovechar el hecho de que
        # Python puede usar índices negativos en listas.

        for l in range(2, self.num_layers): #Itera sobre todas las capas de la red neuronal, excepto la capa de entrada (l comienza en 2 porque estamos considerando las capas ocultas y la capa de salida).
            z = zs[-l]  # Obtiene el vector de activación ponderada para la capa l desde el final de la lista zs
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):

        """
        Evaluamos la precisión de la red neuronal con el conjunto de datos de prueba y nos regresa el número total de predicciones acertadas
        """
        
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):

        """
        Esto proporciona la derivada de la función de costo utilizada en la red neuronal con respecto a las activaciones de salida 
        """
    
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):

    #Calcula y retorna la función sigmoide.
 
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):

    #Calcula y retorna la derivada de la función sigmoide.

    return sigmoid(z)*(1-sigmoid(z))