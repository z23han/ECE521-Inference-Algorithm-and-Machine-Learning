import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

## run algorithm with K=1, 2, 3, 4, 5 and get the percentage of points belonging to each cluster
## scatter plot of the data


data = None
centroids = None
data_num = 0
centroids_num = 0
data_dim = 0
tf_data = None
tf_centroids = None


# initialize data and centroids, and other variables
def init_data(inputFile, K):
    global data, centroids, data_num, data_dim, centroids_num, tf_data, tf_centroids
    # initialize data and centroids
    data = np.float32( np.load(inputFile))
    data = (data - data.mean()) / data.std()
    # update data_num and centroids_num
    data_num, data_dim = data.shape
    centroids_num = K
    #indices = np.array(range(data_num))
    #np.random.shuffle(indices)
    #c_indices = indices[:K]
    centroids = tf.truncated_normal(shape=[centroids_num, data_dim])
    # update tf_data and tf_centroids
    tf_data = tf.placeholder(tf.float32, shape=[data_num, data_dim])
    tf_centroids = tf.Variable(tf.convert_to_tensor(centroids, dtype=tf.float32))
    # get the euclidean distance
    tf_dist = euclidean_dist()
    # get the min index for data set
    tf_min_index = tf.argmin(tf_dist, dimension=1)
    # loss and optimizer
    tf_loss = tf.reduce_sum(tf.reduce_min(euclidean_dist(), 1, keep_dims=True))
    tf_opt = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(tf_loss)
    return tf_min_index, tf_loss, tf_opt


# train the model
def model_train(K):
    inputFile = 'data2D.npy'
    ## initialize session
    graph = tf.Graph()
    with graph.as_default():
        tf_min_index, tf_loss, tf_opt = init_data(inputFile, K)
    with tf.Session(graph=graph) as sess:
        init = tf.initialize_all_variables()
        init.run()
    
        ## model training
        epoch = 500
        losses = []
        
        for i in range(epoch):
            feed_dict = {tf_data: data}
            opt, loss, min_index, centroids = sess.run([tf_opt, tf_loss, tf_min_index, tf_centroids], feed_dict=feed_dict)
            losses.append(loss)
            if not i % 50.:
                print('epoch:', i)
                print('loss:', loss)
                print('centroids:', centroids)
        #plot_model(np.array(range(epoch)), losses, K)
    # get the percentages for different clusters
    data_perc = data_percentage(min_index)
    # get the indices of data in different clusters
    data_indices = mark_data(min_index, K)
    scatter_plot(data_indices, K)
    return data_perc, loss


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
    assert len(min_index) == len(data)
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
def plot_model(updates, loss, K):
    assert len(updates) == len(loss)
    plt.figure(1)
    plt.plot(updates, loss)
    plt.ylabel('loss'), plt.xlabel('updates #')
    plt.title(str(K)+' clusters')
    plt.show()


# calculate euclidean distances
def euclidean_dist():
    # squared values
    tf_data_2 = tf.reduce_sum(tf.square(tf_data), 1)
    tf_centroids_2 = tf.reduce_sum(tf.square(tf_centroids), 1)
    # multiplication 
    tf_mult_sum = tf.matmul(tf_data, tf_centroids, transpose_b=True)
    # definition result 
    tf_res = -2 * tf_mult_sum + tf_centroids_2
    tf_res = tf.transpose(tf_res)
    tf_data_2 = tf.reshape(tf_data_2, [1, data_num])
    tf_res += tf_data_2
    tf_res = tf.transpose(tf_res)
    return tf_res


if __name__ == '__main__':
    # compute different K and report the data point percentage
    the_file_name = 'q1_percentage.txt'
    try:
        os.remove(the_file_name)
    except OSError:
        pass
    the_file = open(the_file_name, 'a')
    losses = []
    K_range = range(1, 6)
    for K in K_range:
        data_perc, loss = model_train(K)
        the_file.write(str(data_perc)+'\n')
        losses.append(loss)
    plot_loss_K(K_range, losses)
    the_file.close()
    


