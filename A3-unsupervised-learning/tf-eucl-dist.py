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
    print(type(tf_data), tf_data)
    print(type(tf_centroids), tf_centroids)
    # squared values
    tf_data_2 = tf.reduce_sum(tf.square(tf_data), 1)
    tf_centroids_2 = tf.reduce_sum(tf.square(tf_centroids), 1)
    # summed values
    tf_mult_sum = tf.matmul(tf_data, tf_centroids, transpose_b=True)
    # res
    tf_res = -2 * tf_mult_sum + tf_centroids_2
    tf_res = tf.transpose(tf_res)
    tf_data_2 = tf.reshape(tf_data_2, [1, data_dim])
    tf_res += tf_data_2
    tf_res = tf.transpose(tf_res)
    return tf_res


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
    tf_dist = ed_dist(sess, data, centroids)
    tf_min = tf.argmin(tf_dist, dimension=1)
    tf_loss = tf.reduce_sum(tf.reduce_mean(ed_dist(sess, data, centroids),1,keep_dims=True))
    s = sess.run(tf_dist)
    m = sess.run(tf_min)
    l = sess.run(tf_loss)
    print(s)
    print(m)
    print(l)

