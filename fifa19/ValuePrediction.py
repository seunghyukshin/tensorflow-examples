import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 단위: €1K (1000유로)
# Load File

data = np.loadtxt('data_modify.csv', delimiter=',', dtype=np.float32, encoding="utf-8")
#data = MinMaxScaler().fit_transform(data)

data_age = data[:, [0]]
data_overall = data[:, [1]]
data_poten = data[:, [2]]
data_value = data[:, [-1]]
data_value = np.log10(data_value)
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
# valid_data_x = MinMaxScaler().fit_transform(valid_data_x)
# valid_data_y = MinMaxScaler().fit_transform(valid_data_y)

test_data_x = data[12000:, [1]]
# just comparing
test_data_y = data[12000:, [-1]]
# test_data_x = MinMaxScaler().fit_transform(test_data_x)
# test_data_y = MinMaxScaler().fit_transform(test_data_y)

# Build Graph
X = tf.placeholder(tf.float32, shape=[None, 1])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([1, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Training
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
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
                                          feed_dict={X: data_poten, Y: data_value})
    if step % 10 == 0:
        print(step, "Cost:", cost_val, "W", w_val, "\n Hypothesis:\n", hy_val)

# Test
# print("Test ",sess.run(hypothesis, feed_dict={X:valid_data_x}))
test_model = [[80], [54]]
#test_model = MinMaxScaler().fit_transform(test_model)
y = sess.run(hypothesis, feed_dict={X: test_model})
print(10**y)
# y_inverse = y_scaler.inverse_transform(y)
# print(y_inverse)
# print("Test ",MinMaxScaler().inverse_transform(sess.run(hypothesis, feed_dict={X: [[26, 92, 93]]})))
# print("Test ", sess.run(hypothesis, feed_dict={X: [[39, 52, 52]]}))

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

ax1.plot(data_age, data_poten-data_overall,color='red')
ax1.set_xlabel('age')
ax1.set_ylabel('potential-overall')
#ax1.plot(data_overall, sess.run(W) * train_data_x + sess.run(b))
ax2.plot(data_overall, data_value, 'ro')
ax2.set_xlabel('overall')
ax2.set_ylabel('value')
#ax2.plot(train_data_x, sess.run(W) * train_data_x + sess.run(b))
ax3.plot(data_poten, data_value, 'ro')
ax3.set_xlabel('poten')
ax3.set_ylabel('value')
ax3.plot(data_poten, sess.run(W) * data_poten + sess.run(b))

# plt.legend()

plt.show()
