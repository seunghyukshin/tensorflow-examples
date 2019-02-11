import tensorflow as tf
import numpy as np
import pandas as pd  # https://dandyrilla.github.io/2017-08-12/pandas-10min/#ch12

# setosa 50개, versicolor 50개, virginica 50개
# training set      25, 25, 25  75
# validation set    10, 10, 10  30
# test set          15, 15, 15  45

data = np.loadtxt('iris_modify_modify.csv', delimiter=',', dtype=np.float32)
mod_data = np.loadtxt('iris_modify.csv', delimiter=',', dtype=np.float32)
mod_x_train = np.stack([mod_data[:25, :-1],mod_data[50:75, :-1],mod_data[100:125,:-1]])
mod_x_train = np.reshape(mod_x_train, newshape=[-1,4])
#print(mod_x_train)

# pd_data = pd.read_csv('iris(150).csv')
# print(pd_data.shape)
# pd_data = pd_data.drop('caseno',axis=1) # 첫번째 열 삭제
# print(pd_data.shape)
'''
x1=data[:,[0]]
x2=data[:,[1]]
x3=data[:,[2]]
x4=data[:,[3]]
'''
x_train = data[:75, :-1]
y_train = data[:75, [-1]]
# pd_x_train = pd_data.loc

x_valid = data[75:105, :-1]
y_valid = data[75:105, [-1]]

x_test = data[105:, :-1]
y_test = data[105:, [-1]]

nb_classes = 3  # 0, 1, 2
keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.int32, [None, 1])  # (?,1)
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])  # (?,7)

W1 = tf.get_variable("W1", shape=[4, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]), name='bias')
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]), name='bias')
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256]), name='bias')
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]), name='bias')
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([256]), name='bias')
L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

W6 = tf.get_variable("W6", shape=[256, nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([nb_classes]), name='bias')
hypothesis = tf.matmul(L5, W6) + b6

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y_one_hot))
train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_predicton = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predicton, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10000):
        sess.run(train, feed_dict={X: x_train, Y: y_train, keep_prob: 0.7})
        if step % 200 == 0:
            loss, acc = sess.run([cost, accuracy],
                                 feed_dict={X: x_train, Y: y_train, keep_prob: 0.7 })
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc
            ))
    print("\n# Validation")
    valid_pred = sess.run(prediction, feed_dict={X: x_valid, keep_prob: 1})
    false_count = 0
    for p, y in zip(valid_pred, y_valid.flatten()):
        false_check = (p == int(y))
        if not false_check:
            false_count += 1
        print("[{}] Prediction: {} True Y: {}".format(false_check, p, int(y)))

    print("\tFalse count: ", false_count)

    print("\n# Test")
    test_pred = sess.run(prediction, feed_dict={X: x_test, keep_prob: 1})
    false_count = 0
    for p, y in zip(test_pred, y_test.flatten()):
        false_check = (p == int(y))
        if not false_check:
            false_count += 1
        print("[{}] Prediction: {} True Y: {}".format(false_check, p, int(y)))
    print("\tFalse count: ", false_count)

    #print("\n# Test : Empty value")
    #test_empty = sess.run(prediction, feed_dict={X:[[5.7, 3.8, None, 0.3],  # expected:0
    #                                                [7.7, None, 6.1, 2.3],  # expected:2
    #                                                [6.3, 3.8, 5.7, 0.3]    # expected:1
    #                                                ],
    #                                             keep_prob:1})
    #print(test_empty)
