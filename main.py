from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from lenet5 import Lenet5

mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot=True)

X_train, y_train = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels

print("Input shape: {}".format(X_train[0].shape))
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

# 0-Padding for LeNet-5's input size
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print("New Input shape: {}".format(X_train[0].shape))

lenet_network = Lenet5(X_train,y_train,X_test,y_test,X_validation,y_validation)
accuracy = lenet_network.train(epochs=2,batch_size=200,auto_save=True)
print("Accuracy on test set: {:.3f}".format(accuracy))

'''
# TODO: some refactoring for restoring the model
lenet_network_restored = Lenet5(X_train, y_train, X_test, y_test, X_validation, y_validation)
lenet_network_restored.restore_model(path='tmp/model.ckpt')
'''