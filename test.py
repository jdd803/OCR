import tensorflow as tf
import numpy as np

def add(a):
    print('asssasa')
    return a,a*2

a = tf.convert_to_tensor([1,2])
b,b1 = add(a)
sess = tf.Session()
with sess.as_default():
    print(sess.run((b,b1)))

c = b1*2
with sess.as_default():
    print(sess.run(c))
