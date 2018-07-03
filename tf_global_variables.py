import tensorflow as tf
import numpy as np
#var = tf.Variable(dtype=tf.int32, shape=None)
var = tf.Variable(0.0)
add_op = tf.add(var, 1.0)
update_op = tf.assign(var, add_op)

mat_var =  tf.Variable(np.zeros(shape=[4,4], dtype=np.float32))
mat_update = tf.assign(mat_var, tf.add(mat_var, np.identity(mat_var.shape[0], dtype=np.float32)))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #sess.run(var, feed_dict={var:0.0})
    for _ in range(4):
        sess.run(update_op)
        print sess.run(var)

###################################################################################################################
    sess.run(mat_update)
    print sess.run(mat_var)
