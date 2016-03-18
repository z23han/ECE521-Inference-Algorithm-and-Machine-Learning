import numpy as np
import tensorflow as tf



## mat1 with (AxD), mat2 with (BxD)
## input numpy arrays, return tensorflow variables
def euclidean_dist(mat1, mat2):
    row1, col1 = mat1.shape
    row2, col2 = mat2.shape
    assert col1 == col2, "matrix1 and matrix2 should have the same dimension"
    tf_mat1 = tf.convert_to_tensor(mat1, dtype=tf.int32)
    tf_mat2 = tf.convert_to_tensor(mat2, dtype=tf.int32)
    tf_rep_mat1 = tf.reshape(tf.tile(tf_mat1, [row2,1]), [row2, row1, col1])
    tf_rep_mat2 = tf.reshape(tf.tile(tf_mat2, [1,row1]), [row2, row1, col2])
    sum_squares = tf.square(tf.sub(tf_rep_mat1, tf_rep_mat2))
    return sum_squares

mat1 = np.array([[1,2], [3,4], [5,6]])
mat2 = np.array([[11,12], [13,14]])


sum_squares = euclidean_dist(mat1, mat2)

sess = tf.Session()
s = sess.run(sum_squares)

print s

