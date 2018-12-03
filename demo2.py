import tensorflow as tf

x1=tf.placeholder(dtype=tf.float32, shape=None)
y1=tf.placeholder(dtype=tf.float32, shape=None)
z1=tf.add(x1, y1)

x2=tf.placeholder(dtype=tf.float32, shape=[1,2])
y2=tf.placeholder(dtype=tf.float32, shape=[2,1])
z2=tf.matmul(x2,y2)

with tf.Session() as sess:
    res1=sess.run(z1, feed_dict={x1:1, y1:2})
    res2=sess.run(z2, feed_dict={x2:[[2,2]], y2:[[3],[3]]})
    print res1
    print res2
