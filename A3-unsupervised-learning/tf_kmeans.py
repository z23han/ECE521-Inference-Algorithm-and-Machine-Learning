import tensorflow as tf
import numpy as np


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
    # update data_num and centroids_num
    data_num, data_dim = data.shape
    centroids_num = K
    indices = np.array(range(data_num))
    np.random.shuffle(indices)
    c_indices = indices[:K]
    centroids = np.float32([data[i] for i in c_indices])
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
def model_train():
    inputFile = 'data2D.npy'
    K = 4
    ## initialize session
    graph = tf.Graph()
    with graph.as_default():
        tf_min_index, tf_loss, tf_opt = init_data(inputFile, K)
    with tf.Session(graph=graph) as sess:
        init = tf.initialize_all_variables()
        init.run()
    
        ## model training
        epoch = 500
        for i in range(epoch):
            feed_dict = {tf_data: data}
            opt, loss, min_index, centroids = sess.run([tf_opt, tf_loss, tf_min_index, tf_centroids], feed_dict=feed_dict)
            if not i % 50.:
                print('epoch:', i)
                print('loss:', loss)
                print('centroids:', centroids)
    


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
    model_train()


