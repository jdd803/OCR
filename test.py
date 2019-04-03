import tensorflow as tf

a = [3]
b = 3.0
a = tf.convert_to_tensor(a)

sess = tf.Session()
print(sess.run(a))
c = a*b
print(sess.run(c))