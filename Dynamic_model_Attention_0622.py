import keras.layers
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Permute, Input, Reshape, Multiply, Add,Attention, Softmax, Lambda
import matplotlib.pyplot as plt
import keras.backend as K
"""

产生一组尺寸为(n,timesteps,input_dim)的随机数据，其中第attention_column个时间序列的值与标签相同，相当于标签只取决于该行(个，行，列)数据

"""
def  get_data(n, time_steps, input_dim, attention_column=10):
    x = np.random.standard_normal(size=(n, time_steps, input_dim)) #标准正态分布随机特征值
    y = np.random.randint(low=0, high=2, size=(n, 1)) #二分类，随机标签值
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim)) #将第attention_column列的值赋值为标签值
    return x, y

n = 50  #样本数
time_steps =  24  #时序时长
input_dim =  6  #通道数
attention_column =  10  #把该行设置为与标签一致
x,y = get_data(n,time_steps,input_dim) #生成数据



class Attention_LSTM():
    def __init__(self,batch_size = 50,time_step = 36,input_dims = 7, output_dims = 2):
        self.batch_size = batch_size
        self.time_step = time_step
        self.input_dims = input_dims
        self.output_dims = output_dims


    def Costom_attention(self,inputs):
        a = Permute((2, 1), name='Transpose_acceleration')(inputs)  ## shape = (batch_size, input_dim, time_steps)
        a_query = Dense(time_steps, activation='softmax', name='Acquire_query')(
            a)  # shape = (batch_size, input_dim, time_steps)
        # 正常的输出应该是batch_size, input_dim, time_steps, 如果不先进行坐标轴变换，就会输出 batch_size, time_steps, time_steps
        # 这里一定首先得进行维度变换，变换后的Dense层才是在每一个序列内部进行q、k、v的计算

        a_keys = Dense(time_steps, name='Acquire_key')(a)  # shape = (batch_size, input_dim, time_steps)
        a_values = Dense(time_steps, name='Acquire_value')(a)  # shape = (batch_size, input_dim, time_steps)

        a_query = Permute((2, 1), name='Query_resume')(a_query)  # shape = (batch_size, time_step, input_dim)
        a_keys = Permute((2, 1), name='Key_resume')(a_keys)  # shape = (batch_size, time_step, input_dim)
        a_values = Permute((2, 1), name='Value_resume')(a_values)  # shape = (batch_size, time_step, input_dim)

        att = Attention(use_scale=True, name='Relevance_mat')([a_query, a_keys, a_values])
        return att

    def model_attention_applied_before_lstm(self,inputs, batch_size=50):
        # a = Permute((2, 1), name='Transpose_acceleration')(inputs)  ## shape = (batch_size, input_dim, time_steps)
        # a_query = Dense(time_steps, activation='softmax', name='Acquire_query')(a)  # shape = (batch_size, input_dim, time_steps)
        # # 正常的输出应该是batch_size, input_dim, time_steps, 如果不先进行坐标轴变换，就会输出 batch_size, time_steps, time_steps
        # # 这里一定首先得进行维度变换，变换后的Dense层才是在每一个序列内部进行q、k、v的计算
        #
        # a_keys = Dense(time_steps,name = 'Acquire_key')(a)  # shape = (batch_size, input_dim, time_steps)
        # a_values = Dense(time_steps, name='Acquire_value')(a)  # shape = (batch_size, input_dim, time_steps)
        #
        # a_query = Permute((2, 1), name='Query_resume')(a_query)  # shape = (batch_size, time_step, input_dim)
        # a_keys = Permute((2, 1), name='Key_resume')(a_keys)  # shape = (batch_size, time_step, input_dim)
        # a_values = Permute((2, 1), name='Value_resume')(a_values)  # shape = (batch_size, time_step, input_dim)

        # att = Attention(use_scale = True,name = 'Attention_built_in')([a_query,a_keys,a_values])
        dV = tf.concat([tf.zeros([batch_size, 1]), inputs[:, -1, 3:4]], axis=1)
        att = self.Costom_attention(inputs)

        # attention_mul = Multiply(name = 'Attention_mechanism')([inputs,a_query]) # shape = (batch_size, time_step, input_dim)
        stepping = LSTM(32, return_sequences=True, name='LSTM_layer_1')(att)

        stepping = LSTM(16, return_sequences=False, name='LSTM_layer_2')(stepping)
        outputs = Dense(2, activation='sigmoid', name='Conclude')(stepping)
        residual = outputs + inputs[:, -1, 1:3] + dV
        return residual

    def get_model(self):
        input_data = Input(shape = (time_steps,input_dim),name = 'data_input')
        label_data = self.model_attention_applied_before_lstm(input_data,batch_size=n)

        model = Model(input_data,label_data)
        # model.compile(optimizer='adam', loss = 'binary_crossentropy',metrics=['accuracy'])
        return model

model = Attention_LSTM()
model = model.get_model()


# print(model(x).shape)
# model.summary()
#
# from keras.utils import plot_model
# plot_model(model,to_file='Cache/model2.png',dpi = 300)
