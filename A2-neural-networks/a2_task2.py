import numpy as np
import tensorflow as tf
from six.moves import range
import matplotlib.pyplot as plt
import sys, os, datetime


image_size=28
num_labels=10
num_hi_units=1000
eta=0.0005
batch_size =50
num_steps =301
num_epochs=500


## logger class used for storing console print output
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

task_dir = 'task2'
if not os.path.exists(task_dir):
    os.makedirs(task_dir)
cur_time = datetime.datetime.now().strftime('_%m_%d_%H%M%S')
sys.stdout = Logger(task_dir + '/task2'+cur_time+'.log')


# load the data
with np.load('notMNIST.npz') as data:
    images, labels = data['images'], data['labels']

data_size = len(labels)

# Reformat into a shape that's more adapted to the models we're going to train.
def reformat(dataset, labels):
    dataset = dataset.reshape((image_size * image_size, -1)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:None]).astype(np.float32)
    return dataset, labels
images_dataset0, labels_dataset = reformat(images, labels)
images_dataset=np.transpose(images_dataset0)


# Divide the dataset into training dataset, validation dataset and testing dataset.
train_dataset, train_labels = images_dataset[:15000,:], labels_dataset[:15000,:]
valid_dataset, valid_labels = images_dataset[15000:16000,:], labels_dataset[15000:16000,:]
test_dataset, test_labels = images_dataset[16000:,:], labels_dataset[16000:,:]


### define the model for neural networks
def model(tf_val, l1_weights, l1_biases, l2_weights, l2_biases):
    logits1 = tf.matmul(tf_val/255., l1_weights) + l1_biases
    hidden = tf.nn.relu(logits1)
    return tf.matmul(hidden, l2_weights) + l2_biases


### get the model log-likelihood
def modelEval(l1_weights, l1_biases, l2_weights, l2_biases, tf_dataset, tf_labels):
    logits = model(tf_dataset, l1_weights, l1_biases, l2_weights, l2_biases)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels))
    likelihood = -1 * loss
    predictions = tf.nn.softmax(logits)
    return logits, predictions, loss, likelihood



# using stochastic gradient descent training with momentum
graph = tf.Graph()
with graph.as_default():
# Input data. For the training data, we use a placeholder that will be fed
# at run time with a training minibatch.
    x = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
    y = tf.placeholder(tf.float32, shape=(None, num_labels))

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_hi_units],stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([num_hi_units]))
    layer2_weights = tf.Variable(tf.truncated_normal([num_hi_units, num_labels],stddev=0.1))
    layer2_biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.(one hidden layer with RELU active function)
    train_logits, train_pred, train_loss, train_log = modelEval(layer1_weights, layer1_biases, layer2_weights, layer2_biases, x, y)
    
    # Optimizer.
    optimizer= tf.train.GradientDescentOptimizer(learning_rate=eta).minimize(train_loss)

    # Predictions for the validation, and test data.
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)
    tf_test_dataset = tf.constant(test_dataset)
    tf_test_labels = tf.constant(test_labels)
    valid_logits, valid_pred, valid_loss, valid_log = modelEval(layer1_weights, layer1_biases, layer2_weights, layer2_biases, tf_valid_dataset, tf_valid_labels)
    test_logits, test_pred, test_loss, test_log = modelEval(layer1_weights, layer1_biases, layer2_weights, layer2_biases, tf_test_dataset, tf_test_labels)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


## calculate errors
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



train_errors = []
train_logs = []
valid_errors = []
valid_logs = []

count = 0
error_cnt = 0
prev_e = 0

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')

    for epoch in range(num_epochs):
        for step in range(num_steps):
	    # Pick an offset within the training data, which has been randomized.
	    # Note: we could use b randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
             # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {x : batch_data, y : batch_labels}
            _ = session.run([optimizer], feed_dict=feed_dict)
       

        ## training
        feed_dict = {x : train_dataset, y : train_labels}
        train_l, train_ll, train_predictions = session.run([train_loss, train_log, train_pred], feed_dict=feed_dict)
        train_acc = accuracy(train_predictions, train_labels)
        train_e = getErrors(train_predictions, train_labels)

        ## validation
        feed_dict = {x : valid_dataset, y : valid_labels}
        valid_l, valid_ll, valid_predictions = session.run([valid_loss, valid_log, valid_pred], feed_dict=feed_dict)
        valid_acc = accuracy(valid_predictions, valid_labels)
        valid_e = getErrors(valid_predictions, valid_labels)

        ## testing
        feed_dict = {x : test_dataset, y : test_labels}
        test_l, test_ll, test_predictions = session.run([test_loss, test_log, test_pred], feed_dict=feed_dict)
        test_acc = accuracy(test_predictions, test_labels)
        test_e = getErrors(test_predictions, test_labels)

        ## test the overfitting and decide if early stopping
        if valid_e >= prev_e:
            error_cnt += 1
        else:
            error_cnt = 0

        ## early stop when more than 10 times increments
        if error_cnt == 10:
            print('Early stop!! epoch %d' % epoch)
            break

        prev_e = valid_e
        count += 1

        train_errors.append(train_e)
        train_logs.append(train_ll)
        valid_errors.append(valid_e)
        valid_logs.append(valid_ll)


	if epoch % 20 ==0:
            print('Epoch %d' % epoch)
            print('Train Loss: %f, Validation Loss: %f' % (train_l, valid_l))
            print('Train Log-Likelihood: %f, Valid Log-Likelihood: %f' % (train_ll, valid_ll))
            print('Train Accuracy: %.1f%%, Valid Accuracy: %.1f%%' % (train_acc, valid_acc))
    print('Test accuracy: %.1f%%' % test_acc)
    print('Test log-likelihood: %f' % test_ll)
    print('Test errors: %d' % test_e)


dataPlot(train_logs, valid_logs, list(range(count)), 'Log-likelihood')
dataPlot(train_errors, valid_errors, list(range(count)), 'Errors')



