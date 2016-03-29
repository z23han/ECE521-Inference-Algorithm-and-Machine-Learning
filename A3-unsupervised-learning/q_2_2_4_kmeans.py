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
            percentage_dict[assignments[i]] += 1. /  10000

    for i in range(k):
        print('the percentage of the data points belonging to the cluster %d: %.2f%%' % (i, percentage_dict[i] * 100))

    
def model_train(k):
    data = np.float32(np.load('data100D.npy'))
    sample_num = data.shape[0]
    dim = data.shape[1]
    cluster = k

    tf_data = tf.placeholder(tf.float32, shape=(sample_num, dim))
    tf_centroids = tf.Variable(tf.truncated_normal([k, dim], mean=0.0, stddev=1.0))
    tf_min_index = tf.argmin(eucl_distance(tf_data, tf_centroids), dimension = 1)
    tf_loss = tf.reduce_sum(tf.reduce_min(eucl_distance(tf_data, tf_centroids),1,keep_dims=True))
    optimizer = tf.train.AdamOptimizer(0.01,0.9,0.99,1e-5).minimize(tf_loss)

    sess = tf.InteractiveSession()

    init = tf.initialize_all_variables()
    init.run()

    epoch = 1000
    loss_list = []
    for i in range(epoch):
        feed_dict = {tf_data: data}
        _, loss, assignments, centroids = sess.run([optimizer, tf_loss, tf_min_index, tf_centroids], feed_dict = feed_dict)
        loss_list.append(loss)
        if (i % 50== 0):
            print("Loss at step %d: %f" % (i, loss))

    cal_percentage(assignments, k)

    plt.title('the loss vs the number of updates 100-D')
    plt.xlabel('the number of updates')
    plt.ylabel('the value of the loss')
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
    return loss



def main():
    ks = [1, 2, 3, 4, 5]
    losses = []
    for k in ks:
        loss = model_train(k)
        losses.append(loss)
    plt.plot(ks, losses, 'o-')
    plt.ylabel('value of loss')
    plt.xlabel('k')
    plt.show()

    


if __name__ == '__main__':
    main()


