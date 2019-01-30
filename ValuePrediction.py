import tensorflow as tf
import numpy as np

# 단위: €1K (1000유로)
# Load File
data = np.loadtxt('data_modify.csv', delimiter=',', dtype=np.float32, encoding="utf-8")
train_data_x = data[:9000, :3]  # [[age, overall, potential], ...]
train_data_y = data[:9000, [-1]]  # [[value], ...]

valid_data_x = data[9000:12000, :3]
valid_data_y = data[9000:12000, [-1]]

test_data_x = data[12000:, :3]
# just comparing
test_data_y = data[12000:, [-1]]

# Build Graph
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Training
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-12)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    w_val, cost_val, hy_val, _ = sess.run([W,cost, hypothesis, train],
                                   feed_dict={X: train_data_x, Y: train_data_y})
    if step % 10 == 0:
        print(step, "Cost:", cost_val,"W",w_val, "\n Hypothesis:\n", hy_val)

# Test
print("Test ",sess.run(hypothesis, feed_dict={X:valid_data_x}))