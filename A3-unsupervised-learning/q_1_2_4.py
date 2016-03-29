import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_result(data, k, clusters, assignments):
    num_sample, dim = data.shape

    mark = ['or', 'ob', 'og', 'ok', 'oc']

    for i in range(num_sample):
        plt.plot(data[i, 0], data[i, 1], mark[assignments[i]])

    mark = ['Dr', 'Db', 'Dg', 'Dk', 'Dc']

    for i in range(len(clusters)):
        plt.plot(clusters[i, 0], clusters[i, 1], mark[i])


def eucl_distance(samples, centroids):
    expanded_vectors = tf.expand_dims(samples, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(
        tf.square(tf.sub(expanded_vectors,
                         expanded_centroids)), 2)
    distances = tf.transpose(distances)
    return distances

def cal_percentage(assignments, k):
    percentage_dict = {}
    for i in range(len(assignments)):
        if assignments[i] not in percentage_dict:
            percentage_dict[assignments[i]] = 1. / 10000
        else:
            percentage_dict[assignments[i]] += 1. / 10000

    for i in range(k):
        print('the percentage of the data points belonging to the cluster %d: %.2f%%' % (i, percentage_dict[i] * 100))

    
def main(k):
    data = np.float32(np.load('data2D.npy'))[: 10000 * 2 / 3, :]
    val_data = np.float32(np.load('data2D.npy'))[10000 - 10000 * 2 / 3: , :]
    sample_num = data.shape[0]
    val_sample_num = val_data.shape[0]
    dim = data.shape[1]
    cluster = k

    with tf.name_scope('training_process'):
        tf_data = tf.placeholder(tf.float32, shape=(sample_num, dim))
        tf_centroids = tf.Variable(tf.truncated_normal([k, dim], mean=0.0, stddev=1.0))
        tf_min_index = tf.argmin(eucl_distance(tf_data, tf_centroids), dimension = 1)
        tf_loss = tf.reduce_sum(tf.reduce_min(eucl_distance(tf_data, tf_centroids),1,keep_dims=True))
        optimizer = tf.train.AdamOptimizer(0.01,0.9,0.99,1e-5).minimize(tf_loss)

    with tf.name_scope('valid_process'):
        tf_val_data = tf.placeholder(tf.float32, shape=(val_sample_num, dim))
        tf_val_loss = tf.reduce_sum(tf.reduce_min(eucl_distance(tf_val_data, tf_centroids),1,keep_dims=True))

    sess = tf.InteractiveSession()

    init = tf.initialize_all_variables()
    init.run()

    epoch = 800
    loss_list = []
    val_loss_list = []
    count = 0
    for i in range(epoch):
        feed_dict = {tf_data: data, tf_val_data: val_data}
        _, loss, assignments, centroids, val_loss = sess.run([optimizer, tf_loss, tf_min_index, tf_centroids, tf_val_loss], feed_dict = feed_dict)
        loss_list.append(loss)
        val_loss_list.append(val_loss)
        # early stop
        if i > 0:
            if val_loss_list[-1] > val_loss_list[-2]:
                count = count + 1
            elif count == 10:
                break
            else:
                count = 0
            
        if (i % 50== 0):
            print("Loss at step %d: %f" % (i, loss))

    cal_percentage(assignments, k)

    print('the loss of the validation data is: %f' % val_loss_list[-1])
    print('the loss of the training data is: %f' % loss_list[-1])
    print (count)

    plt.figure(1)
    plt.plot(range(len(loss_list)), loss_list, 'r')
    plt.plot(range(len(val_loss_list)), val_loss_list, 'b')
    plt.legend(['training loss', 'validation loss'])
    plt.title('training loss vs validation loss')
    plt.xlabel('updates')
    plt.ylabel('loss')
    plt.show()

    return loss, val_loss

if __name__ == '__main__':
    ks = [1, 2, 3, 4, 5]
    losses = []
    valid_losses = []
    for k in ks:
        loss, valid_loss = main(k)
        losses.append(loss)
        valid_losses.append(valid_loss)
    plt.figure(1)
    plt.subplot(121)
    plt.plot(range(1, 6), losses, 'ro-')
    plt.xlabel('training ' + str(k)+' clusters')
    plt.ylabel('losses')
    plt.subplot(122)
    plt.plot(range(1, 6), valid_losses, 'bo-')
    plt.xlabel('validation ' + str(k) + ' clusters')
    plt.ylabel('losses')
    plt.show()
    


