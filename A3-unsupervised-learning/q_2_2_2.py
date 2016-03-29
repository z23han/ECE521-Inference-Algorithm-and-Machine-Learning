import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils
import math

def eucl_distance(samples, centroids):
    expanded_vectors = tf.expand_dims(samples, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(
        tf.square(tf.sub(expanded_vectors,
                         expanded_centroids)), 2)
    distances = tf.transpose(distances)
    return distances

def plot_result(data, k, assignments):
    num_sample, dim = data.shape

    mark = ['or', 'ob', 'og', 'ok', '^r']

    for i in range(num_sample):
        plt.plot(data[i, 0], data[i, 1], mark[assignments[i]])


data2D = np.float32(np.load('data2D.npy'))
data = (data2D - data2D.mean()) / data2D.std()
#data = np.float32(np.load('data2D.npy'))
k = 3

num_sample = data.shape[0]
dim = data.shape[1]

tf_mean = tf.Variable(tf.random_normal([k, dim], mean=0.0, stddev=1.0, dtype=tf.float32))
#tf_mean = tf.Variable(tf.random_uniform([k, dim], minval=-3, maxval=3, dtype=tf.float32))
tf_covariance = tf.Variable(0.5 * tf.exp(tf.random_normal([k], mean=0.0, stddev=1.0, dtype=tf.float32)))
#phi = tf.Variable(tf.random_normal([1, k], mean=0.0, stddev=1.0, dtype=tf.float32))
phi = tf.Variable(tf.truncated_normal([1, k], mean=0.0, stddev=1.0, dtype=tf.float32))
log_pi = utils.logsoftmax(phi)

#tf_data = tf.Variable(data)
tf_data = tf.placeholder(tf.float32, shape=(num_sample, dim))

tf_expanded_data = tf.expand_dims(tf_data, 0)
tf_expanded_mean = tf.expand_dims(tf_mean, 1)

tf_sub = tf.sub(tf_expanded_data, tf_expanded_mean)
tf_sub_square = tf.square(tf_sub)
tf_sub_square_sum = tf.reduce_sum(tf_sub_square, 2, True)
tf_sub_square_sum_02 = tf.squeeze(tf.transpose(tf_sub_square_sum))
tf_index = (-0.5) * tf.div(tf_sub_square_sum_02, tf_covariance)
tf_log_second_term = tf_index
tf_log_first_term = (-0.5 * dim) * tf.log(2 * math.pi * tf_covariance)

# log(P(x|z))
tf_log_x_gan_z = tf.add(tf_log_first_term, tf_log_second_term)
# log(P(x,z))
tf_log_pro_z_x_gan_z = tf.add(log_pi, tf_log_x_gan_z)

#tf_pro_z_x_gan_z = tf.exp(tf_log_pro_z_x_gan_z)
#tf_sum_pro_z_x_gan_z = utils.reduce_logsumexp(tf_pro_z_x_gan_z, 1)
#tf_log_sum_pro_z_x_gan_z = tf.log(tf_sum_pro_z_x_gan_z)

tf_log_sum_pro_z_x_gan_z = utils.reduce_logsumexp(tf_log_pro_z_x_gan_z, 1)
#tf_log_like = tf.reduce_mean(tf_log_sum_pro_z_x_gan_z)
tf_log_like = tf.reduce_sum(tf_log_sum_pro_z_x_gan_z)
tf_loss = -1 * tf_log_like

optimizer = tf.train.AdamOptimizer(0.01, 0.9, 0.99, 1e-5).minimize(tf_loss)
#optimizer = tf.train.AdamOptimizer(0.001).minimize(tf_loss)
#optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(tf_loss)

#tf_assignments = tf.argmax(tf_log_pro_z_x_gan_z, 1)
#tf_assignments = tf.argmax(tf_log_x_gan_z, 1)
#tf_assignments = tf.argmin(eucl_distance(tf_data, tf_mean), dimension = 1)


tf_assignments_01 = tf.transpose(tf.expand_dims(tf.reduce_sum(tf_log_pro_z_x_gan_z, 1), 0))
tf_assignments_02 = tf.sub(tf_log_pro_z_x_gan_z, tf_assignments_01)
tf_assignments = tf.argmax(tf_assignments_02, 1)


sess = tf.InteractiveSession()

init = tf.initialize_all_variables()
init.run()

loss_list = []
for i in range(3000):
    feed_dict = {tf_data: data}
    _, loss, assignments, mean, covariance, log_pii = sess.run([optimizer, tf_loss, tf_assignments, tf_mean, tf_covariance, log_pi], feed_dict = feed_dict)
    loss_list.append(loss)
    if (i % 50== 0):
        print("Loss at step %d: %f" % (i, loss))


plt.subplot(121)
plt.plot(range(len(loss_list)), loss_list)
plt.xlabel('updates')
plt.ylabel('loss')
plt.subplot(122)
plot_result(data, k, assignments)
plt.title('scattered plot')
plt.show()

import os

the_file_name = 'q_2_1_2_param.txt'
try:
    os.remove(the_file_name)
except OSError:
    pass

the_file = open(the_file_name, 'a')

the_file.write('mean: ' + str(mean) + '\n')
the_file.write('covariance: ' + str(covariance) + '\n')
the_file.write('learning rate: ' + str(0.01) + '\n')
the_file.write('log_pi: ' + str(log_pii) + '\n')

the_file.close()


'''
print('log_pi')
print(log_pi.eval())
print('tf_mean')
print(tf_mean.eval())
print('tf_expanded_mean')
print(tf_expanded_mean.eval())
print('tf_covariance')
print(tf_covariance.eval())
print('tf_index')
print(tf_index.eval())
print('tf_log_first_term')
print(tf_log_first_term.eval())
print('tf_log_second_term')
print(tf_log_second_term.eval())
print('tf_log_x_gan_z')
print(tf_log_x_gan_z.eval())
print('tf_log_pro_z_x_gan_z')
print(tf_log_pro_z_x_gan_z.eval())
print('tf_log_sum_pro_z_x_gan_z')
print(tf_log_sum_pro_z_x_gan_z.eval())
print('tf_loss')
print(tf_loss.eval())
'''

