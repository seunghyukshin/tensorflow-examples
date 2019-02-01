import tensorflow as tf
import numpy as np

# setosa 50개, versicolor 50개, virginica 50개
# training set      25, 25, 25
# validation set    10, 10, 10
# test set          15, 15, 15

data = np.loadtxt('iris_modify_modify.csv', delimiter=',', dtype=np.float32)

x_train = data[:75, :-1]
y_train = data[:75, [-1]]

x_valid = data[75:105, :-1]
y_valid = data[75:105, [-1]]

x_test = data[105:, :-1]
y_test = data[105:, [-1]]
# print(x_test, x_test, len(x_test))
nb_classes = 3  # 0, 1, 2

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost)

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(train, feed_dict={X: x_train, Y: y_train})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_train, Y: y_train}))
    # Validation

    #run_valid = sess.run(hypothesis, feed_dict={X: x_valid})
    #print("\n",run_valid, sess.run(tf.argmax(run_valid, 1)))
