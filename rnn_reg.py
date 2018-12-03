import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TIME_STEP = 10       # rnn time step
INPUT_SIZE = 1      # rnn input size
CELL_SIZE = 32      # rnn cell size
LR = 0.02           # learning rate

steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps); y_np = np.cos(steps)    # float32 for converting torch FloatTensor
plt.plot(steps, y_np, 'r-', label='target (cos)'); plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best'); plt.show()

tf_x = tf.placeholder(dtype=tf.float32, shape=[None, TIME_STEP, INPUT_SIZE])
tf_y = tf.placeholder(dtype=tf.float32, shape=[None, TIME_STEP, INPUT_SIZE])

rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=CELL_SIZE)
init_s = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
outputs, final_s = tf.nn.dynamic_rnn(rnn_cell, tf_x, initial_state=init_s, time_major=False)

outs2D = tf.reshape(outputs, [-1, CELL_SIZE])
net_outs2D = tf.layers.dense(outs2D, INPUT_SIZE)
outs = tf.reshape(net_outs2D, [-1, TIME_STEP, INPUT_SIZE])

loss = tf.losses.mean_squared_error(labels=tf_y, predictions=outs)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

plt.figure(1, figsize=(12, 5)); plt.ion()       # continuously plot

for step in range(60):
    start, end = step * np.pi, (step+1)*np.pi
    steps = np.linspace(start, end, TIME_STEP)
    x = np.sin(steps)[np.newaxis, :, np.newaxis]
    y = np.cos(steps)[np.newaxis, :, np.newaxis]
    if 'final_s_' not in globals():
        feed_dict = {tf_x:x, tf_y:y}
    else:
        feed_dict = {tf_x:x, tf_y:y, init_s:final_s_}
    _, pred_, final_s_ = sess.run([train_op, outs, final_s], feed_dict)

    # plotting
    plt.plot(steps, y.flatten(), 'r-'); plt.plot(steps, pred_.flatten(), 'b-')
    plt.ylim((-1.2, 1.2)); plt.draw(); plt.pause(0.05)
plt.ioff(); plt.show()

