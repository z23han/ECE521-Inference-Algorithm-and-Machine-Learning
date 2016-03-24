import tensorflow as tf
import numpy as np


# use tensorflow broadcast
# input: data, centroids
# output: distances
def ed_dist(sess, data, centroids):
    # get dimensions
    data_dim, _ = data.shape
    centroids_dim, _ = centroids.shape
    # convert to tensorflow const
    tf_data = tf.convert_to_tensor(data, dtype=tf.float64)
    tf_centroids = tf.convert_to_tensor(centroids, dtype=tf.float64)
    # squared values
    tf_data_2 = tf.reduce_sum(tf.square(tf_data), 1)
    tf_centroids_2 = tf.reduce_sum(tf.square(tf_centroids), 1)
    # summed values
    tf_mult_sum = tf.matmul(tf_data, tf_centroids, transpose_b=True)
    # centroids transpose
    tf.transpose(tf_centroids_2)
    # res
    res = -2 * tf_mult_sum + tf_centroids_2
    res = tf.transpose(res)
    tf_data_2 = tf.reshape(tf_data_2, [1, data_dim])
    res += tf_data_2
    res = tf.transpose(res)
    return res


def ed_dist2(data, centroids):
    row1, col1 = data.shape
    row2, col2 = centroids.shape
    tf_data = tf.convert_to_tensor(data, dtype=tf.float64)
    tf_centroids = tf.convert_to_tensor(centroids, dtype=tf.float64)
    tf_rep_data = tf.reshape(tf.tile(tf_data, [row2, 1]), [row2, row1, col1])
    tf_rep_centroids = tf.reshape(tf.tile(tf_centroids, [1, row1]), [row2, row1, col2])
    res = tf.reduce_sum(tf.square(tf.sub(tf_rep_data, tf_rep_centroids)), [2])
    return res


if __name__ == '__main__':
    sess = tf.Session()
    data = np.random.rand(5, 2)
    centroids = data[:3]
    dist1 = ed_dist(sess, data, centroids)
    dist2 = ed_dist2(data, centroids)
    s = sess.run(dist1)
    s2 = sess.run(dist2)
    print(s)
    print(s2)

