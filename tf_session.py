import tensorflow as tf
#######################SESSION AND TENSORFLOW#############################################################
#######################SESSION AND TENSORFLOW#############################################################
#######################SESSION AND TENSORFLOW#############################################################
print '#######################SESSION AND TENSORFLOW#############################################################'
m1 = tf.constant([[2,3],[4,5]])
m2 = tf.constant([[4], [7]])

dot_operation = tf.matmul(m1, m2)

sess = tf.Session()
print sess.run(dot_operation)
sess.close()

#Alternatively one can also do

with tf.Session() as sess:
    print sess.run(dot_operation)
#######################PLACEHOLDER########################################################################
#######################PLACEHOLDER########################################################################
#######################PLACEHOLDER########################################################################
print '#######################PLACEHOLDER########################################################################'

x1 = tf.placeholder(dtype=tf.float32, shape=None)
x2 = tf.placeholder(dtype=tf.float32, shape=None)
sum = x1 + x2

m1 = tf.placeholder(dtype=tf.float64, shape=[2,2])
m2 = tf.placeholder(dtype=tf.float64, shape=[2,1])
M = tf.matmul(m1, m2)

with tf.Session() as sess:
    sum_act = sess.run(sum, feed_dict={x1:3.4, x2:4.5})
    print sum_act
    M_act = sess.run(M, feed_dict={m1:[[2,3],[4,5]], m2:[[2],[6]]})
    print M_act

