import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from tqdm import tqdm,trange

class Lenet5():

    def __init__(self,train_data,train_labels,test_data,test_labels,validation_data=None,validation_labels=None,mean=0,stddev=0.3,learning_rate=0.001):

        self.train_data = train_data
        self.train_labels = train_labels
        assert(len(self.train_labels)==len(self.train_data))
        assert (self.train_data[0].shape[0] == 32 and self.train_data[0].shape[1] == 32)

        self.validation_data=validation_data
        self.validation_labels=validation_labels
        assert (len(self.validation_labels) == len(self.validation_data))
        assert (self.validation_data[0].shape[0] == 32 and self.validation_data[0].shape[1] == 32)

        self.test_data=test_data
        self.test_labels=test_labels
        assert (len(self.test_data) == len(self.test_labels))
        assert (self.test_data[0].shape[0] == 32 and self.test_data[0].shape[1] == 32)

        self.num_outputs = 10

        self.X = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
        self.y = tf.placeholder(tf.int32, shape=(None,self.num_outputs))

        self.mu = mean
        self.sigma = stddev

        # Layer 1: Input 32x32x1, Output 28x28x6
        self.conv1_kernels = tf.Variable(tf.truncated_normal(shape=[5,5,1,6],mean=self.mu,stddev=self.sigma))
        self.conv1_biases = tf.Variable(tf.zeros(6))
        self.conv1 = tf.nn.conv2d(self.X,self.conv1_kernels,[1,1,1,1],padding='VALID') + self.conv1_biases
        # Activation
        self.conv1 = tf.nn.relu(self.conv1)
        # Pooling -> from 28x28 to 14x14
        self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')


        # Layer 2: Input 14x14x6, Output 10x10x16
        self.conv2_kernels = tf.Variable(tf.truncated_normal(shape=[5,5,6,16],mean=self.mu,stddev=self.sigma))
        self.conv2_biases = tf.Variable(tf.zeros(16))
        self.conv2 = tf.nn.conv2d(self.pool1,self.conv2_kernels,[1,1,1,1],padding='VALID') + self.conv2_biases
        # Activation 2
        self.conv2 = tf.nn.relu(self.conv2)
        # Pooling -> from 10x10x16 to 5x5x16
        self.pool2 = tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

        # Flatten -> from 5x5x16 to 400x1
        self.flattened = flatten(self.pool2)

        # Fully Connected Layer n.1
        self.fcl1_weights = tf.Variable(tf.truncated_normal(shape=[400,120],mean=self.mu,stddev=self.sigma))
        self.fcl1_biases = tf.Variable(tf.zeros(120))
        self.fcl1 = tf.matmul(self.flattened,self.fcl1_weights) + self.fcl1_biases
        # Activation 3
        self.fcl1 = tf.nn.relu(self.fcl1)

        # Fully Connected Layer n.2
        self.fcl2_weights = tf.Variable(tf.truncated_normal(shape=[120, 84], mean=self.mu, stddev=self.sigma))
        self.fcl2_biases = tf.Variable(tf.zeros(84))
        self.fcl2 = tf.matmul(self.fcl1, self.fcl2_weights) + self.fcl2_biases
        # Activation 4
        self.fcl2 = tf.nn.relu(self.fcl2)

        # Fully Connected Layer n.3
        self.fcl3_weights = tf.Variable(tf.truncated_normal(shape=[84,10],mean=self.mu,stddev=self.sigma))
        self.fcl3_biases = tf.Variable(tf.zeros(10))
        self.logits = tf.matmul(self.fcl2, self.fcl3_weights) + self.fcl3_biases

        # Loss and metrics
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,labels=self.y)
        self.loss_op = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.training_step = self.optimizer.minimize(self.loss_op)

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.saver = tf.train.Saver()



    def train(self,epochs,batch_size):
        assert (epochs > 0 and batch_size > 0)

        num_examples = len(self.train_data)

        print('Training the model . . .')

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            total_steps = trange(epochs)
            for epoch in total_steps:
                self.train_data,self.train_labels = shuffle(self.train_data, self.train_labels)
                for offset in xrange(0,num_examples, batch_size):
                    end = offset + batch_size
                    X_batch,y_batch = self.train_data[offset:end], self.train_labels[offset:end]

                    _,acc,cross = session.run([self.training_step,self.accuracy_operation,self.cross_entropy], feed_dict={self.X: X_batch, self.y: y_batch})

                if self.validation_data != None:
                    validation_accuracy = self.evaluate(self.validation_data, self.validation_labels, batch_size)
                    #print("Epoch {} - validation accuracy {:.3f} ".format(epoch+1,validation_accuracy))
                    total_steps.set_description("Epoch {} - validation accuracy {:.3f} ".format(epoch+1,validation_accuracy))


    def evaluate(self,X_data, y_data,batch_size):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
            accuracy = sess.run(self.accuracy_operation, feed_dict={self.X: batch_x, self.y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples