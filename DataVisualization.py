import mnist_loader
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
import network
from PIL import Image

#leemos dato de mnist
tr_d, va_d, te_d = mnist_loader.load_data()

v=tr_d[0][0] # primera imagen
print(tr_d[1][0]) #su valor
im=np.reshape(v,(28,28)) #la acomodamos para visualizar
plt.imshow(im) # la visualizamos
plt.show()
v=np.reshape(im,(784,1)) #La reacomodamos como matriz de (784,1) para poderla meter a la red

imtest=imread('Numero3.jpg') #Leemos nuestra imagen
plt.imshow(imtest) #La visualizamos
imtest=np.reshape(imtest,(784,3)) # La convertimos en vector
#Convertimos a blanco y negro la imagen:
lst = []
for i in imtest:
    pix=i[0]*0.2125+i[1]*0.7174+i[2]*0.0721 #transfomamos a escala de grises
    if(pix<125):
        pix=255. #Como la hoja es blanca y el papel negro, lo negro lo ponemos con mayor luminosidad
    else:
        pix=0. #lo blanco lo ponemos como negro
    lst.append(pix)
imtest=np.array(lst).reshape(28,28) #acomodamos la imagen para poder ver como quedó
imtest=(imtest/imtest.max()) #normalizamos
plt.imshow(imtest) #visualizamos la imagen
print("imagen a reconocer")
plt.show()

test_v=np.reshape(imtest,(784,1)) #La acomodamos como matrix (784,1) para poder usarla en la red

#cargamos nuestra red:
archivo_lectura = open("red_prueba1.pkl",'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()

#evaluamos nuestra imagen en la red:
a=net.feedforward(test_v)
print(a)
max = np.where(a == np.amax(a)) #encontramos el valor maximo
print("La red reconoce la imagen como un:")
print(max[0]) #predicción de la red


print("Ahora probamos con la imagen del 5 de la base de datos:")
a1=net.feedforward(v)
print(a1)
max = np.where(a1 == np.amax(a1))
print("La red reconoce la imagen como un:")
print(max[0])
