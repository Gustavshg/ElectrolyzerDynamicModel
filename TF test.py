import tensorflow as tf

#import tensorflow_addons as tfa

#tfa_crf文件是tfa.text中关于crf的一个文件
#可直接拿来用，不用安装整个tfa

from tensorflow.keras.layers import Layer,LSTM,Bidirectional,Embedding,Input,Dense,Dropout
from tensorflow.keras.initializers import glorot_uniform
import numpy as np
class BiLSTMCRF(tf.keras.models.Model):
    def __init__(self,vocab_size,embed_size,units,num_tags,*args,**kwargs):
        super(BiLSTMCRF,self).__init__()
        self.num_tags = num_tags
        self.embedding = Embedding(input_dim = vocab_size, output_dim = embed_size)
        self.bilstm = Bidirectional(LSTM(units,return_sequences = True), merge_mode = 'concat')
        self.dense = Dense(num_tags)
        self.dropout = Dropout(0.5)

    def call(self,inputs):
        inputs_length = tf.math.reduce_sum(tf.cast(tf.math.not_equal(inputs,0),dtype = tf.int32),axis = -1)
        x = self.embedding(inputs)
        x = self.bilstm(x)
        x = self.dropout(x)
        logits = self.dense(x)

        return logits , inputs_length

    def loss(self,logits, targets, inputs_length):
        targets = tf.cast(targets,dtype= tf.int32)
        log_likelihood = 1
        return log_likelihood

    def build(self,input_shape):
        shape = tf.TensorShape([self.num_tags,self.num_tags])
        self.transition_params = self.add_weight(name = 'transition_params', shape = shape, initializer = glorot_uniform)
        super(BiLSTMCRF,self).build(input_shape)
def test():
    batch_size = 16
    vocab_size = 100
    max_seq_length = 8
    embed_size = 32
    units = 13
    num_tags = 50




if __name__ == '__main__':
    print(1)