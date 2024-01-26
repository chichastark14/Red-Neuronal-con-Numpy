import mnist_loader
import network

training_data , test_data, _ = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
net.SGD(training_data, 90, 10, 0.01, test_data=test_data)
with open('miRed.pkl','wb') as file1:
	pickle.dump(net,file1)

file1=open('miRed.pkl','rb')
net2 = pickle.load(file1)

exit()
a=aplana(Imagen)
resultado = net.feedforward(a)
print(resultado)