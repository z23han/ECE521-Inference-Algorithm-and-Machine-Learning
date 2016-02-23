import numpy as np
import tensorflow as tf
from six.moves import range
import matplotlib.pyplot as plt
import sys, os, datetime


image_size = 28
num_labels = 10
data_size = 0


## logger class used for storing console print output
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)



## obtain training, validation, and testing data
def getData():
    ## load the data
    global data_size
    with np.load('notMNIST.npz') as data:
        images, labels = data['images'], data['labels']
    data_size = len(labels)
    ## reformat the data
    images_dataset0, labels_dataset = reformat(images, labels)
    images_dataset = np.transpose(images_dataset0)
    
    ## Divide the dataset into training dataset, validation dataset and testing dataset.
    train_dataset, train_labels = images_dataset[:15000,:], labels_dataset[:15000,:]
    valid_dataset, valid_labels = images_dataset[15000:16000,:], labels_dataset[15000:16000,:]
    test_dataset, test_labels = images_dataset[16000:,:], labels_dataset[16000:,:]

    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels



## Reformat into a shape that's more adapted to the models we're going to train.
def reformat(dataset, labels):
    global image_size, num_labels
    dataset = dataset.reshape((image_size * image_size, -1)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:None]).astype(np.float32)
    return dataset, labels



## build softmax model
def buildModel(tf_train_dataset, tf_train_labels, learning_rate=0.001, momentum=0.5):
    global image_size, num_labels
    ## variable 
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels], stddev=0.1))
    biases = tf.Variable(tf.zeros([num_labels]))
    ## model
    logits, prediction, loss, likelihood = softmaxModel(tf_train_dataset, tf_train_labels, weights, biases)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss)

    return weights, biases, prediction, loss, likelihood, optimizer



## softmax model
def softmaxModel(tf_dataset, tf_labels, weights, biases):
    global image_size, num_labels
    logits = tf.matmul(tf_dataset/255., weights) + biases
    prediction = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels))
    likelihood = -1 * loss
    return logits, prediction, loss, likelihood



## calculate accuracy
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


## get number of errors
def getErrors(predictions, labels):
    return np.sum(np.argmax(predictions, 1) != np.argmax(labels, 1))



## plot training vs epoch && validation vs epoch
def dataPlot(train, valid, epochs, y_label):
    assert len(train) == len(valid)
    assert len(train) == len(epochs)
    l1, = plt.plot(epochs, train, 'r')
    l2, = plt.plot(epochs, valid, 'g')
    plt.xlabel('epoch')
    plt.ylabel(y_label)
    if y_label == 'Log-likelihood':
        plt.legend(['Training '+y_label, 'Validation '+y_label], loc='lower right')
    else:
        plt.legend(['Training '+y_label, 'Validation '+y_label], loc='upper center')
    plt.show()



## train the model
def modelTraining(*data):
    global image_size, num_labels, data_size
    ## grab the data for training, validation and testing
    train_dataset = data[0]
    train_labels = data[1]
    valid_dataset = data[2]
    valid_labels = data[3]
    test_dataset = data[4]
    test_labels = data[5]
    batch_size = 150

    ## first we need to build the graph
    graph = tf.Graph()
    with graph.as_default():
        ## build the model for training
        x = tf.placeholder(tf.float32, shape=[None, image_size*image_size])
        y = tf.placeholder(tf.float32, shape=[None, num_labels])
        weights, biases, train_pred, train_loss, train_likelihood, train_opt = buildModel(x, y)
        ## build the validation and testing model
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_valid_labels = tf.constant(valid_labels)
        valid_logits, valid_pred, valid_loss, valid_likelihood = softmaxModel(tf_valid_dataset, tf_valid_labels, weights, biases)
        tf_test_dataset = tf.constant(test_dataset)
        tf_test_labels = tf.constant(test_labels)
        test_logits, test_pred, test_loss, test_likelihood = softmaxModel(tf_test_dataset, tf_test_labels, weights, biases)

    num_steps = 100
    num_epochs = 501

    train_errors = []
    train_logs = []
    valid_errors = []
    valid_logs = []


    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")

        for epoch in range(num_epochs):
            for step in range(num_steps):
                ## pick an offset within the training data, randomized
                ## Note: we could use better randomization across epochs
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                ## generate minibatch
                batch_data = train_dataset[offset: (offset + batch_size), :]
                batch_labels = train_labels[offset: (offset + batch_size), :]
                ## train
                feed_dict = {x: batch_data, y: batch_labels}
                _ = session.run([train_opt], feed_dict=feed_dict)

            ## training
            feed_dict = {x: train_dataset, y: train_labels}
            train_l, train_log, train_predictions = session.run([train_loss, train_likelihood, train_pred], feed_dict=feed_dict)
            train_accuracy = accuracy(train_predictions, train_labels)
            train_e = getErrors(train_predictions, train_labels)
            train_errors.append(train_e)
            train_logs.append(train_log)
            ## validation
            feed_dict = {x: valid_dataset, y: valid_labels}
            valid_l, valid_log, valid_predictions = session.run([valid_loss, valid_likelihood, valid_pred], feed_dict=feed_dict)
            valid_accuracy = accuracy(valid_predictions, valid_labels)
            valid_e = getErrors(valid_predictions, valid_labels)
            valid_errors.append(valid_e)
            valid_logs.append(valid_log)
            ## testing
            feed_dict = {x: test_dataset, y: test_labels}
            test_l, test_log, test_predictions = session.run([test_loss, test_likelihood, test_pred], feed_dict=feed_dict)
            test_accuracy = accuracy(test_predictions, test_labels)
            test_e = getErrors(test_predictions, test_labels)
            if epoch % 50 == 0:
                print("epoch %d" % epoch)
                print("Train Loss: %f, Valid Loss: %f" % (train_l, valid_l))
                print("Train LogLikelihood: %f, Valid LogLikelihood: %f" % (train_log, valid_log))
                print("Train accuracy: %.1f%%, Valid accuracy :%.1f%%" % (train_accuracy, valid_accuracy))
                print("Train errors: %d, Valid errors: %d" % (train_e, valid_e))
        print("Test LogLikelihood; %f" % test_log)
        print("Test accuracy: %.1f%%" % test_accuracy)
        print("Test errors: %d" % test_e)


    dataPlot(train_logs, valid_logs, list(range(num_epochs)), 'Log-likelihood')
    dataPlot(train_errors, valid_errors, list(range(num_epochs)), 'Errors')





if __name__ == '__main__':
    task_dir = 'task1'
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    cur_time = datetime.datetime.now().strftime('_%m_%d_%H%M%S')
    sys.stdout = Logger(task_dir+'/task1'+cur_time+'.log')
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = getData()
    modelTraining(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)

