#Moreno Santiago José Miguel
#Fecha: 17 de febrero del 2024
#Actualización: LLamamos los nuevos métodos que son fundamentales para SGD

import mnist_loader
import network
import pickle #Importamos esta librería, tras ejecutar el codigo vimos que daba error por falta de esta libreria

training_data , test_data, _ = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
net.set_momentum_rate(0.9)
net.initialize_velocities()
net.SGD(training_data, 90, 10, 0.01, test_data=test_data)


exit() 


with open('miRed.pkl','wb') as file1:
	pickle.dump(net,file1)

file1=open('miRed.pkl','rb')
net2 = pickle.load(file1)



a=aplana(Imagen)
resultado = net.feedforward(a)
print(resultado)
