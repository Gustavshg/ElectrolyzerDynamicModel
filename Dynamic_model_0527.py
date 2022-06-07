import tensorflow as tf
import Dynamic_data_feed_0527

class V_LSTM(tf.keras.models.Model):
    def __init__(self , units = 10):
        super(V_LSTM,self).__init__()
        self.lstm = tf.keras.layers.GRU(units)
        self.dense = tf.keras.layers.Dense(1,activation = 'tanh')

    def build(self,input_shape):
        super(V_LSTM,self).build(input_shape)

    def call(self,inputs):
        x = self.lstm(inputs)
        y = self.dense(x)
        return y

class V_2LSTM(tf.keras.models.Model):
    def __init__(self , units = 20):
        super(V_2LSTM,self).__init__()
        self.lstm1 = tf.keras.layers.GRU(units,return_sequences = True)
        self.lstm2 = tf.keras.layers.GRU(units)
        self.dense = tf.keras.layers.Dense(1)

    def build(self,input_shape):
        super(V_2LSTM,self).build(input_shape)

    def call(self,inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        y = self.dense(x)
        return y

class V_3LSTM(tf.keras.models.Model):
    def __init__(self , units = 20):
        super(V_3LSTM,self).__init__()
        self.lstm1 = tf.keras.layers.GRU(units,return_sequences = True)
        self.lstm2 = tf.keras.layers.GRU(units,return_sequences = True)
        self.lstm3 = tf.keras.layers.GRU(units)
        self.dense = tf.keras.layers.Dense(1)

    def build(self,input_shape):
        super(V_3LSTM,self).build(input_shape)

    def call(self,inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.lstm3(x)
        y = self.dense(x)
        return y

# model_seq = tf.keras.Sequential()
# model_seq.add(tf.keras.layers.Embedding(input_dim = 1000,output_dim = 64))
# model_seq.add(tf.keras.layers.GRU(15,return_sequences = False))
# model_seq.add(tf.keras.layers.Dense(1))
# model_seq.compile()