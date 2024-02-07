#Moreno Santiago José Miguel
#Fecha: 26 de febrero del 2024
#Actualización: Import pickle y Verificación de salida

#Se comprobo que lo unico que hace es mostrar lass epocas de esa manera sabemos que tanto se acerca a 10,000 
#se importo la libreria pickle
#Se modifico la salida para mostrar la prediccion y notamos que la función "aplana" no esta definida en ningun lugar de este u otro código

import mnist_loader
import network
import pickle #Importamos esta librería, tras ejecutar el codigo vimos que daba error por falta de esta libreria
from PIL import Image

training_data , test_data, _ = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
net.SGD(training_data, 10, 10, 0.01, test_data=test_data)
with open('miRed.pkl','wb') as file1:
	pickle.dump(net,file1)

file1=open('miRed.pkl','rb')
net2 = pickle.load(file1)


a=aplana(Imagen)
resultado = net.feedforward(a)
print(resultado)

exit() #Modificar el lugar de la función exit() para que no solo muestre las epocas, si no tambien el resultado