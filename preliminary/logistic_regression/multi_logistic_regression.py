'''
    Multi Logistic Regression
    
Created on 2019. 5. 21.
@author: HRKim
[출처:모두의딥러링]
'''

import tensorflow as tf
import numpy as np

# 실행할 때마다 같은 결과를 출력하기 위한 seed 값 설정
seed =0
np.random.seed(seed)
tf.set_random_seed(seed)


# x,y의 데이터 값
# tensorflow는 행렬형태로 계산을 하기 때문에  np.array를 적용하여 배열을 행렬로 변환
x_data = np.array([[2, 3],[4, 3],[6, 4],[8, 6],[10, 7],[12, 8],[14, 9]])
y_data = np.array([0, 0, 0, 1, 1, 1,1]).reshape(7, 1)

# 입력 값을 placeholder에 저장
# placeholder( dtype, shape=None, name=None)
# Its value must be fed using the feed_dict optional argument to Session.run().

X = tf.placeholder(tf.float64, shape=[None, 2])
Y = tf.placeholder(tf.float64, shape=[None, 1])

# 기울기 a,   bias b의 값을 임의로 정함.
# x의 값이 2개임으로 random_uniform에 [2,1]로 지정
a = tf.Variable(tf.random_uniform([2,1], dtype=tf.float64)) # [2,1] 의미: 들어오는 값은 2개, 나가는 값은 1개
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))

# HYPOTHESIS: y 시그모이드 함수의 방정식을 세움
# y = 1/( 1+ np.e**( a * x_data + b))
y = tf.sigmoid(tf.matmul(X, a) + b)

# COST FUNCTION: 오차를 구하는 함수
loss = -tf.reduce_mean(Y * tf.log(y) + (1 - Y) * tf.log(1 - y))

# LEARNING RATE 학습률 값
learning_rate=0.1

# 오차를 최소로 하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# tf.cast : Tensor를 새로운 형태로 캐스팅하는데 사용
#   tf.cast( y>0.5, dtype=tf.float64)
#    y의 값이 0.5보다 크면 true임으로 float64 type으로 1. 을 return
#    y의 값이 0.5보다 작으면 true임으로 float64 type으로 0. 을 return
# predicted = tf.cast(y > 0.5, dtype=tf.float64)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))

# 학습
print('{0:=^50}'.format('exploration'))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3001):
        # run( fetches, feed_dict=None, options=None, run_metadata=None)
        # The value returned by run() has the same shape as the fetches argument,
        # where the leaves are replaced by the corresponding values returned by Tensorflow.
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={X: x_data, Y: y_data})
        
        if (i + 1) % 300 == 0:
            print("step=%d, a1=%.4f, a2=%.4f, b=%.4f, loss=%.4f" % (i + 1, a_[0], a_[1], b_, loss_))
            w1 = a_[0]
            w2 = a_[1]
            bi = b_
            

    print('{0:=^50}'.format('exploitation'))
    
    new_x = np.array([7, 6.]).reshape(1, 2)  #[7, 6]은 각각 공부 시간과 과외 수업수.
    new_y = sess.run(y, feed_dict={X: new_x})
    
    print("공부 시간: %d, 개인 과외 수: %d" % (new_x[:,0], new_x[:,1]))
    print("합격 가능성: %6.2f %%" % (new_y*100))
    print('{0:=^50}'.format('End of source'))
            
with tf.Session() as ss: 
    # 어떻게 활용하는가
    #    y = 1/( 1+ np.e**( a * x_data + b))
    #    y = tf.sigmoid(tf.matmul(X, a) + b)    
    print('a1 = ', w1, ' a2 = ' , w2, ' bias = ' ,bi)
    w = np.array([w1,w2])
    yy = tf.sigmoid(tf.matmul(X,w)+bi)
    
    new_yy = ss.run(yy, feed_dict={X: new_x})
    print("합격 가능성: %6.2f %%" % (new_yy*100))
    print('{0:=^50}'.format('실제적용'))