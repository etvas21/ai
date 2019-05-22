'''
  Queue Runners
    다수의 파일에서 데이터를 가져올때 사용
    또는 대용량의 파일을 가져올때 사용
    
Created on 2019. 5. 22.

@author: HRKim
[출처:https://www.youtube.com/redirect?q=http%3A%2F%2Fhunkim.github.io%2Fml%2F&event=video_description&redir_token=WCbfsdvM7AEImx3brf2-bntZQyx8MTU1ODU5MjEzNkAxNTU4NTA1NzM2&v=o2q4QNnoShY]
'''

import tensorflow   as tf

filename_queue = tf.train.string_input_producer(
    ['../data/score.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults =[[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1],xy[-1:]], batch_size=10)
    
X = tf.placeholder(tf.float32, shape=[None,3])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# HYPOTHESIS
hypothesis = tf.matmul( X,W) + b

# Simplified cost/Lost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    for step in range(100):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, hy_val, _ = sess.run(
            [ cost, hypothesis, train], feed_dict={X:x_batch, Y:y_batch})
        
        if step % 2 == 0:
            print(step, 'Cost: ', cost_val, '\nPrediction:\n', hy_val)
coord.requet_stop()
coord.join(threads)

            

    