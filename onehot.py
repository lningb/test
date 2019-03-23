

import tensorflow as tf
import numpy as np

def one_hot(labels):
    sess = tf.Session()
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels,1)
    indices = tf.expand_dims(tf.range(0,batch_size,1),1)
    concated = tf.concat([indices,labels],1)
    onehot_labels = tf.sparse_to_dense(concated,tf.stack([batch_size,10]),1.0,0.0)
    temp = sess.run(onehot_labels)
    return temp

if __name__ == '__main__':
    #labels = np.array([1,3,5,7,9,1,2,3,4,5,6])
    labels = [1,3,5,7,9,1,2,3,4,5,6]
    aa = one_hot(labels)
    print(aa,type(aa))
