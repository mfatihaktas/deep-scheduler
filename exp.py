import numpy as np
import tensorflow as tf

tf.reset_default_graph()
sess = tf.InteractiveSession()

def exp():
  s_len = 3
  N, T = 2, 2
  l = np.arange(1, N*T*s_len+1).reshape((N, T, s_len) )
  x = tf.convert_to_tensor(l, dtype=tf.float32)
  print("x= {}".format(x.eval() ) )
  
  # l = np.arange(1, N*T+1).reshape((N, T) )
  # o = tf.convert_to_tensor(l, dtype=tf.int32)
  # print("o= {}".format(o.eval() ) )
  
  # l = np.array([1] * N*T).reshape((N, T) )
  # l = np.array([1] * N*T).reshape((N, T) )
  l = np.random.randint(s_len, size=(N, T) )
  a = tf.convert_to_tensor(l, dtype=tf.int32)
  print("a= {}".format(a.eval() ) )
  print("a[1:]= {}".format(a[1:].eval() ) )
  
  print("shape(x)= {}".format(tf.shape(x).eval() ) )
  indices = tf.range(0, tf.shape(x)[0]*tf.shape(x)[1] ) * tf.shape(x)[2] + tf.reshape(a, [-1] )
  print("indices= {}".format(indices.eval() ) )
  ro = tf.reshape(tf.gather(tf.reshape(x, [-1] ), indices), (N, T) )
  print("ro= {}".format(ro.eval() ) )

if __name__ == "__main__":
  exp()