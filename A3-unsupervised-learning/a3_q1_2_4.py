import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

## run algorithm with K=1, 2, 3, 4, 5 and use 2/3 data as training and 1/3 data as validation data


training_data = None
validation_data = None
centroids = None
training_num = 0
centroids_num = 0
data_dim = 0
tf_data_set = None
tf_centroids = None


# initialize data and centroids, and other variables
def init_data(inputFile, K):
    global training_data, validation_data, centroids, training_num, data_dim, centroids_num 
    global tf_data_set, tf_centroids
    # initialize data and centroids
    data = np.float32( np.load(inputFile))
    data = (data - data.mean()) / data.std()
    # update data_num and centroids_num
    data_num, data_dim = data.shape
    centroids_num = K
    # training data and validation data
    training_num = int(2./3 * data_num)
    training_data = data[:training_num]
    validation_data = data[training_num:]
    centroids = tf.truncated_normal(shape=[centroids_num, data_dim])
    # update tf_data_set and tf_centroids
    tf_data_set = tf.placeholder(tf.float32, shape=[None, data_dim])
    tf_centroids = tf.Variable(tf.convert_to_tensor(centroids, dtype=tf.float32))
    ########### for the training cases #####################
    # get the euclidean distance
    tf_train_dist = euclidean_dist(tf_data_set, tf_centroids, training_num, centroids_num)
    # get the min index for data set
    tf_train_min_index = tf.argmin(tf_train_dist, dimension=1)
    # loss and optimizer
    tf_train_loss = tf.reduce_sum(tf.reduce_min(euclidean_dist(tf_data_set, tf_centroids, training_num, centroids_num), 
        1, keep_dims=True))
    tf_train_opt = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(tf_train_loss)
    ########### for the validation cases ####################
    tf_valid_dist = euclidean_dist(tf_data_set, tf_centroids, (data_num-training_num), centroids_num)
    tf_valid_min_index = tf.argmin(tf_valid_dist, dimension=1)
    tf_valid_loss = tf.reduce_sum(tf.reduce_min(euclidean_dist(tf_data_set, tf_centroids, (data_num-training_num), centroids_num), 
        1, keep_dims=True))
    return tf_train_min_index, tf_train_loss, tf_train_opt, tf_valid_loss


# train the model
def model_train(K):
    inputFile = 'data2D.npy'
    ## initialize session
    graph = tf.Graph()
    with graph.as_default():
        tf_train_min_index, tf_train_loss, tf_train_opt, tf_valid_loss = init_data(inputFile, K)
    with tf.Session(graph=graph) as sess:
        init = tf.initialize_all_variables()
        init.run()
    
        ## model training
        epoch = 500
        train_losses = []
        valid_losses = []
        
        for i in range(epoch):
            feed_dict = {tf_data_set: training_data}
            opt = sess.run([tf_train_opt], feed_dict=feed_dict)
            train_loss, train_min_index, centroids = sess.run([tf_train_loss, tf_train_min_index, tf_centroids], feed_dict=feed_dict)
            feed_dict = {tf_data_set: validation_data}
            valid_loss = sess.run(tf_valid_loss, feed_dict=feed_dict)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if not i % 50.:
                print('epoch:', i)
                print('train loss:', train_loss)
                print('centroids:', centroids)
                print('valid loss:', valid_loss)
        plot_model(np.array(range(epoch)), train_losses, K, 'training')
        plot_model(np.array(range(epoch)), valid_losses, K, 'validation')
    # get the indices of data in different clusters
    #data_indices = mark_data(train_min_index, K)
    #scatter_plot(data_indices, K)
    return train_loss, valid_loss


# scatter plot
def scatter_plot(data_indices, K):
    color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    plt.figure(1)
    for i in range(len(data_indices)):
        indices = data_indices[i]
        data_cluster = data[indices]        # get a cluster of data
        x_cluster = data_cluster[:,0]
        y_cluster = data_cluster[:,1]
        plt.plot(x_cluster, y_cluster, color[i]+'o')
    plt.title(str(K)+' clusters')
    plt.show()
    return




# plot loss vs. K
def plot_loss_K(K, loss):
    plt.figure(1)
    plt.plot(K, loss, 'ro-')
    plt.ylabel('loss')
    plt.xlabel('K')
    plt.show()



# mark the points based on the min_index
def mark_data(min_index, K):
    assert len(min_index) == training_num
    # index_bucket used for storing indices with the same index
    index_bucket = []
    for c in range(K):
        c_indices = np.where(min_index == c)
        index_bucket.append(c_indices[0])
    return index_bucket



# compute the data point percentage
def data_percentage(min_index):
    count_dict = {}
    perc_dict = {}
    for index in min_index:
        if index not in count_dict:
            count_dict[index] = 1
        else:
            count_dict[index] += 1
    total_num = len(min_index)
    assert total_num == data_num
    for k, val in count_dict.iteritems():
        perc_dict[k] = val/float(total_num)
    return perc_dict



# plot loss vs. number of updates
def plot_model(updates, loss, K, data_type):
    assert len(updates) == len(loss)
    plt.figure(1)
    plt.plot(updates, loss)
    plt.ylabel('loss'), plt.xlabel('updates #')
    plt.title(data_type + ': ' + str(K)+' clusters')
    plt.show()


# calculate euclidean distances
def euclidean_dist(tf_input1, tf_input2, input_num1, input_num2):
    # squared values
    tf_input1_sq = tf.reduce_sum(tf.square(tf_input1), 1)
    tf_input2_sq = tf.reduce_sum(tf.square(tf_input2), 1)
    # multiplication 
    tf_mult_sum = tf.matmul(tf_input1, tf_input2, transpose_b=True)
    # definition result 
    tf_res = -2 * tf_mult_sum + tf_input2_sq
    tf_res = tf.transpose(tf_res)
    tf_data_2 = tf.reshape(tf_input1_sq, [1, input_num1])
    tf_res += tf_data_2
    tf_res = tf.transpose(tf_res)
    return tf_res


if __name__ == '__main__':
    K_range = range(1, 6)
    train_losses = []
    valid_losses = []
    for K in K_range:
        train_loss, valid_loss = model_train(K)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
    the_file_name = 'q1_loss.txt'
    try:
        os.remove(the_file_name)
    except OSError:
        pass
    the_file = open(the_file_name, 'a')
    for k in K_range:
        the_file.write(str(k)+' clusters\n')
        the_file.write('training: '+str(train_losses[k-1]))
        the_file.write('    ')
        the_file.write('validation: '+str(valid_losses[k-1]))
        the_file.write('\n')
    the_file.close()

    


