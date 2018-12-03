#!/usr/bin/python

import sys
import tensorflow as tf

m1=tf.constant([[2,2]])
m2=tf.constant([[3],[3]])

dot=tf.matmul(m1,m2)

with tf.Session() as sess:
    res=sess.run(dot)
    print res
