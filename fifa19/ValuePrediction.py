import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# 단위: €1K (1000유로)
# Load File

data = np.loadtxt('data_modify.csv', delimiter=',', dtype=np.float32, encoding="utf-8")
#data = MinMaxScaler().fit_transform(data)

# fit:내부적으로 normalization 에 필요한 값들을 계산해서 저장
# fit_transform: 실제 각 값들이 normalization
train_data_x = data[:9000, [1]]  # [[age, overall, potential], ...]
train_data_y = data[:9000, [-1]]  # [[value], ...]
'''
train_data_x = MinMaxScaler().fit_transform(train_data_x)
y_scaler = MinMaxScaler()
train_data_y = y_scaler.fit_transform(train_data_y)
'''

valid_data_x = data[9000:12000, [1]]
valid_data_y = data[9000:12000, [-1]]
#valid_data_x = MinMaxScaler().fit_transform(valid_data_x)
#valid_data_y = MinMaxScaler().fit_transform(valid_data_y)

test_data_x = data[12000:, [1]]
# just comparing
test_data_y = data[12000:, [-1]]
#test_data_x = MinMaxScaler().fit_transform(test_data_x)
#test_data_y = MinMaxScaler().fit_transform(test_data_y)

# Build Graph
X = tf.placeholder(tf.float32, shape=[None, 1])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([1,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Training
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
train = optimizer.minimize(cost)
'''
prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    w_val, cost_val, hy_val, _ = sess.run([W, cost, hypothesis, train],
                                          feed_dict={X: train_data_x, Y: train_data_y})
    if step % 10 == 0:
        print(step, "Cost:", cost_val, "W", w_val, "\n Hypothesis:\n", hy_val)

# Test
# print("Test ",sess.run(hypothesis, feed_dict={X:valid_data_x}))
y = sess.run(hypothesis, feed_dict={X: [[80]]})
print(y)
#y_inverse = y_scaler.inverse_transform(y)
#print(y_inverse)
# print("Test ",MinMaxScaler().inverse_transform(sess.run(hypothesis, feed_dict={X: [[26, 92, 93]]})))
# print("Test ", sess.run(hypothesis, feed_dict={X: [[39, 52, 52]]}))

plt.plot(train_data_x, train_data_y, 'ro')
plt.plot(train_data_x, sess.run(W)*train_data_x+sess.run(b))
plt.legend()
plt.show()