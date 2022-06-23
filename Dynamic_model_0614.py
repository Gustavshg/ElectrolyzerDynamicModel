import tensorflow as tf

class V_2LSTM(tf.keras.models.Model):
    def __init__(self , units = 68):
        super(V_2LSTM,self).__init__()
        self.lstm1 = tf.keras.layers.GRU(units,return_sequences = True)
        self.lstm2 = tf.keras.layers.GRU(units)
        self.dense = tf.keras.layers.Dense(2)

    def build(self,input_shape):
        super(V_2LSTM,self).build(input_shape)

    def call(self,inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        y = self.dense(x)
        return y

class V_3LSTM(tf.keras.models.Model):
    def __init__(self , units = 24):
        super(V_3LSTM,self).__init__()
        self.lstm1 = tf.keras.layers.GRU(units,return_sequences = True)
        self.lstm2 = tf.keras.layers.GRU(units,return_sequences = True)
        self.lstm3 = tf.keras.layers.GRU(units)
        self.dense = tf.keras.layers.Dense(2)

    def build(self,input_shape):
        super(V_3LSTM,self).build(input_shape)

    def call(self,inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.lstm3(x)
        y = self.dense(x)
        return y

class V_4LSTM(tf.keras.models.Model):
    def __init__(self , units = 24):
        super(V_4LSTM,self).__init__()
        self.lstm1 = tf.keras.layers.GRU(units,return_sequences = True)
        self.lstm2 = tf.keras.layers.GRU(units,return_sequences = True)
        self.lstm3 = tf.keras.layers.GRU(units, return_sequences=True)
        self.lstm4 = tf.keras.layers.GRU(units)
        self.dense1 = tf.keras.layers.Dense(units)
        self.dense = tf.keras.layers.Dense(2)

    def build(self,input_shape):
        super(V_4LSTM,self).build(input_shape)

    def call(self,inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.lstm4(x)
        x = self.dense1(x)
        y = self.dense(x)
        return y

