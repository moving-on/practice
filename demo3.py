import tensorflow as tf

x1=tf.Variable(0)

add_op=tf.add(x1,1)
update=tf.assign(x1, add_op)

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print sess.run(x1)
