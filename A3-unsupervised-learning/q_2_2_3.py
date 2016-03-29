import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils
import math

def plot_result(data, k, assignments):
    num_sample, dim = data.shape

    mark = ['or', 'ob', 'og', 'ok', 'oc']

    for i in range(num_sample):
        plt.plot(data[i, 0], data[i, 1], mark[assignments[i]])


data2D = np.float32(np.load('data2D.npy')[:10000* 2 / 3, :])
data = (data2D - data2D.mean()) / data2D.std()
data2D_val = np.float32(np.load('data2D.npy')[:10000 - 10000 * 2 / 3, :])
data_val = (data2D_val - data2D_val.mean()) / data2D_val.std()
#data = np.float32(np.load('data2D.npy'))
k = 3

num_sample = data.shape[0]
dim = data.shape[1]

num_sample_val = data_val.shape[0]
dim_val = data_val.shape[1]

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

tf_log_sum_pro_z_x_gan_z = utils.reduce_logsumexp(tf_log_pro_z_x_gan_z, 1)
tf_log_like = tf.reduce_sum(tf_log_sum_pro_z_x_gan_z)
tf_loss = -1 * tf_log_like

optimizer = tf.train.AdamOptimizer(0.01, 0.9, 0.99, 1e-5).minimize(tf_loss)


tf_assignments_01 = tf.transpose(tf.expand_dims(tf.reduce_sum(tf_log_pro_z_x_gan_z, 1), 0))
tf_assignments_02 = tf.sub(tf_log_pro_z_x_gan_z, tf_assignments_01)
tf_assignments = tf.argmax(tf_assignments_02, 1)


# validation data
tf_data_val = tf.placeholder(tf.float32, shape=(num_sample_val, dim_val))
tf_expanded_data_val = tf.expand_dims(tf_data_val, 0)

tf_sub_val = tf.sub(tf_expanded_data_val, tf_expanded_mean)
tf_sub_square_val = tf.square(tf_sub_val)
tf_sub_square_sum_val = tf.reduce_sum(tf_sub_square_val, 2, True)
tf_sub_square_sum_02_val = tf.squeeze(tf.transpose(tf_sub_square_sum_val))
tf_index_val = (-0.5) * tf.div(tf_sub_square_sum_02_val, tf_covariance)
tf_log_second_term_val = tf_index_val
tf_log_first_term_val = (-0.5 * dim) * tf.log(2 * math.pi * tf_covariance)

tf_log_x_gan_z_val = tf.add(tf_log_first_term_val, tf_log_second_term_val)
tf_log_pro_z_x_gan_z_val = tf.add(log_pi, tf_log_x_gan_z_val)

tf_log_sum_pro_z_x_gan_z_val = utils.reduce_logsumexp(tf_log_pro_z_x_gan_z_val, 1)
tf_log_like_val = tf.reduce_sum(tf_log_sum_pro_z_x_gan_z_val)
tf_loss_val = -1 * tf_log_like_val
####################################################################

sess = tf.InteractiveSession()

init = tf.initialize_all_variables()
init.run()

loss_list = []
loss_list_val = []
for i in range(3000):
    feed_dict = {tf_data: data, tf_data_val:data_val}
    _, loss, assignments, mean, loss_val = sess.run([optimizer, tf_loss, tf_assignments, tf_mean, tf_loss_val], feed_dict = feed_dict)
    loss_list.append(loss)
    loss_list_val.append(loss_val)
    if (i % 50== 0):
        print("Loss at step %d: %f" % (i, loss))
        print(mean)
        print("Training Loss at step %d: %f" % (i, loss_list[-1]))
        print("Validation Loss at step %d: %f" % (i, loss_list_val[-1]))


plt.figure(1)
plt.subplot(121)
plt.plot(range(len(loss_list)), loss_list)
plt.xlabel('updates')
plt.title('training loss ' + str(k))

plt.subplot(122)
plt.plot(range(len(loss_list_val)), loss_list_val)
plt.xlabel('updates')
plt.title('validation loss ' + str(k))
plt.show()

plt.figure(2)
plot_result(data, k, assignments)
plt.title('scattered plot ' + str(k) + ' clusters')
plt.show()


'''
print('log_pi')
print(log_pi.eval())
print('tf_data')
print(tf_data.eval())
print('tf_expanded_data')
print(tf_expanded_data.eval())
print('tf_mean')
print(tf_mean.eval())
print('tf_expanded_mean')
print(tf_expanded_mean.eval())
print('tf_covariance')
print(tf_covariance.eval())
print('tf_sub')
print(tf_sub.eval())
print('tf_sub_square')
print(tf_sub_square.eval())
print('tf_sub_square_sum')
print(tf_sub_square_sum.eval())
print('tf_sub_square_sum_02')
print(tf_sub_square_sum_02.eval())
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

