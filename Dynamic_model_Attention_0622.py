import keras.layers
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Permute, Input, Multiply, Add, Attention
import matplotlib.pyplot as plt
import keras.backend as K

"""

产生一组尺寸为(n,timesteps,input_dim)的随机数据，其中第attention_column个时间序列的值与标签相同，相当于标签只取决于该行(个，行，列)数据
这里是模板里面的东西，已经不用了
"""
# def  get_data(n, time_steps, input_dim, attention_column=10):
#     x = np.random.standard_normal(size=(n, time_steps, input_dim)) #标准正态分布随机特征值
#     y = np.random.randint(low=0, high=2, size=(n, 1)) #二分类，随机标签值
#     x[:, attention_column, :] = np.tile(y[:], (1, input_dim)) #将第attention_column列的值赋值为标签值
#     return x, y
#
# n = 50  #样本数
# time_steps =  24  #时序时长
# input_dim =  6  #通道数
# attention_column =  10  #把该行设置为与标签一致
# x,y = get_data(n,time_steps,input_dim) #生成数据


"""
这里是我们编写的模型，在实例化完了之后，还是需要用get_model 才能得到想要的模型
"""


class Attention_LSTM():
    """这个模型使用一层注意力机制，两层LSTM"""

    def __init__(self, time_steps=64, input_dims=7, output_dims=2):
        self.time_steps = time_steps
        self.input_dims = input_dims  # 目前的模型里面应该是7
        self.output_dims = output_dims  # 目前的模型里面只能是2

    def Costom_attention(self, inputs, layer_name='Relevance_mat'):
        a = Permute((2, 1), name='Transpose_acceleration')(inputs)  ## shape = (batch_size, input_dim, time_steps)
        a_query = Dense(self.time_steps, activation='softmax', name='Acquire_query')(
            a)  # shape = (batch_size, input_dim, time_steps)
        # 正常的输出应该是batch_size, input_dim, time_steps, 如果不先进行坐标轴变换，就会输出 batch_size, time_steps, time_steps
        # 这里一定首先得进行维度变换，变换后的Dense层才是在每一个序列内部进行q、k、v的计算

        a_keys = Dense(self.time_steps, name='Acquire_key')(a)  # shape = (batch_size, input_dim, time_steps)
        a_values = Dense(self.time_steps, name='Acquire_value')(a)  # shape = (batch_size, input_dim, time_steps)

        a_query = Permute((2, 1), name='Query_resume')(a_query)  # shape = (batch_size, time_step, input_dim)
        a_keys = Permute((2, 1), name='Key_resume')(a_keys)  # shape = (batch_size, time_step, input_dim)
        a_values = Permute((2, 1), name='Value_resume')(a_values)  # shape = (batch_size, time_step, input_dim)

        att = Attention(use_scale=True, name=layer_name)(
            [a_query, a_keys, a_values])  # shape = (batch_size, time_step, input_dims)
        return att

    def model_attention_applied_before_lstm(self, inputs):
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
        self.batch_size = inputs.shape[0]
        zeros = inputs[:, -1, 2:3] - inputs[:, -1, 2:3]  # 生成一个在虚拟模型状态形状为None, 1 的零向量，作为温度的补齐
        dV = tf.concat(values=[inputs[:, -1, 2:3], zeros], axis=1)  # 序号为2的列是dV_static_star，因此需要补齐
        V_T = inputs[:, -1, 0:2]  # 序号为0，1的列分别为上一时刻的电压、温度
        att = self.Costom_attention(inputs)

        # attention_mul = Multiply(name = 'Attention_mechanism')([inputs,a_query]) # shape = (batch_size, time_step, input_dim)
        stepping = LSTM(32, return_sequences=True, name='LSTM_layer_1')(att)
        stepping = LSTM(16, return_sequences=False, name='LSTM_layer_2')(stepping)
        outputs = Dense(self.output_dims, activation='sigmoid', name='Conclude')(stepping)
        residual = outputs + V_T + dV

        return residual

    def get_model(self):
        input_data = Input(shape=(self.time_steps, self.input_dims), name='data_input')
        output_data = self.model_attention_applied_before_lstm(input_data)

        model = Model(input_data, output_data)

        return model


class Attention_LSTM_double_1():
    def __init__(self, time_steps=64, input_dims=7, output_dims=2):
        self.time_steps = time_steps
        self.input_dims = input_dims  # 目前的模型里面应该是7
        self.output_dims = output_dims  # 目前的模型里面只能是2

    def Costom_attention_1(self, inputs, layer_name='Primary_attention'):
        """这个模型使用两个注意力机制，两层LSTM"""
        a = Permute((2, 1), name='Transpose_acceleration')(inputs)  ## shape = (batch_size, input_dim, time_steps)
        a_query = Dense(self.time_steps, activation='softmax', name='Acquire_query')(
            a)  # shape = (batch_size, input_dim, time_steps)
        # 正常的输出应该是batch_size, input_dim, time_steps, 如果不先进行坐标轴变换，就会输出 batch_size, time_steps, time_steps
        # 这里一定首先得进行维度变换，变换后的Dense层才是在每一个序列内部进行q、k、v的计算

        a_keys = Dense(self.time_steps, name='Acquire_key')(a)  # shape = (batch_size, input_dim, time_steps)
        a_values = Dense(self.time_steps, name='Acquire_value')(a)  # shape = (batch_size, input_dim, time_steps)

        a_query = Permute((2, 1), name='Query_resume')(a_query)  # shape = (batch_size, time_step, input_dim)
        a_keys = Permute((2, 1), name='Key_resume')(a_keys)  # shape = (batch_size, time_step, input_dim)
        a_values = Permute((2, 1), name='Value_resume')(a_values)  # shape = (batch_size, time_step, input_dim)

        att = Attention(use_scale=True, name=layer_name)(
            [a_query, a_keys, a_values])  # shape = (batch_size, time_step, input_dims) #层内已经自带softmax效果
        return att

    def Costom_attention_2(self, inputs, layer_name='Secondary_attention'):
        """这个模型使用两个注意力机制，两层LSTM"""
        a = Permute((2, 1), name='Transpose_acceleration_2')(inputs)  ## shape = (batch_size, input_dim, time_steps)
        a_query = Dense(self.time_steps, activation='softmax', name='Acquire_query_2')(
            a)  # shape = (batch_size, input_dim, time_steps)
        # 正常的输出应该是batch_size, input_dim, time_steps, 如果不先进行坐标轴变换，就会输出 batch_size, time_steps, time_steps
        # 这里一定首先得进行维度变换，变换后的Dense层才是在每一个序列内部进行q、k、v的计算

        a_keys = Dense(self.time_steps, name='Acquire_key_2')(a)  # shape = (batch_size, input_dim, time_steps)
        a_values = Dense(self.time_steps, name='Acquire_value_2')(a)  # shape = (batch_size, input_dim, time_steps)

        a_query = Permute((2, 1), name='Query_resume_2')(a_query)  # shape = (batch_size, time_step, input_dim)
        a_keys = Permute((2, 1), name='Key_resume_2')(a_keys)  # shape = (batch_size, time_step, input_dim)
        a_values = Permute((2, 1), name='Value_resume_2')(a_values)  # shape = (batch_size, time_step, input_dim)

        att = Attention(use_scale=True, name=layer_name)(
            [a_query, a_keys, a_values])  # shape = (batch_size, time_step, input_dims)

        return att

    def model_attention_applied_before_lstm(self, inputs):
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
        zeros = inputs[:, -1, 2:3] - inputs[:, -1, 2:3]  # 生成一个在虚拟模型状态形状为None, 1 的零向量，作为温度的补齐
        dV = tf.concat(values=[inputs[:, -1, 2:3], zeros], axis=1)  # 序号为2的列是dV_static_star，因此需要补齐

        V_T = inputs[:, -1, 0:2]  # 序号为0，1的列分别为上一时刻的电压、温度
        V_ori = inputs[:, -1, 0:1] * 2.5
        T_ori = inputs[:, -1, 1:2] * 100
        V_T = tf.concat(values = [V_ori,T_ori] , axis = 1)
        att = self.Costom_attention_1(inputs, layer_name='Primary_attention')
        att = self.Costom_attention_2(att, layer_name='Secondary_attention')

        # attention_mul = Multiply(name = 'Attention_mechanism')([inputs,a_query]) # shape = (batch_size, time_step, input_dim)
        stepping = LSTM(32, return_sequences=True, name='LSTM_layer_1')(att)
        stepping = LSTM(16, return_sequences=False, name='LSTM_layer_2')(stepping)
        outputs = Dense(self.output_dims, activation='sigmoid', name='Conclude')(stepping)
        residual = outputs + V_T + dV

        return residual

    def get_model(self):
        input_data = Input(shape=(self.time_steps, self.input_dims), name='data_input')
        output_data = self.model_attention_applied_before_lstm(input_data)

        model = Model(input_data, output_data)
        # model.compile(optimizer='adam', loss = 'binary_crossentropy',metrics=['accuracy'])
        return model

class Attention_LSTM_double_simplified():
    def __init__(self, time_steps=64, input_dims=7, output_dims=2):
        self.time_steps = time_steps
        self.input_dims = input_dims  # 目前的模型里面应该是7
        self.output_dims = output_dims  # 目前的模型里面只能是2

    def Costom_attention_1(self, inputs, layer_name='Primary_attention'):
        """这个模型使用两个注意力机制，两层LSTM"""
        a = Permute((2, 1), name='Transpose_acceleration')(inputs)  ## shape = (batch_size, input_dim, time_steps)
        a_query = Dense(self.time_steps, activation='softmax', name='Acquire_query')(
            a)  # shape = (batch_size, input_dim, time_steps)
        # 正常的输出应该是batch_size, input_dim, time_steps, 如果不先进行坐标轴变换，就会输出 batch_size, time_steps, time_steps
        # 这里一定首先得进行维度变换，变换后的Dense层才是在每一个序列内部进行q、k、v的计算

        a_keys = Dense(self.time_steps, name='Acquire_key')(a)  # shape = (batch_size, input_dim, time_steps)
        # a_values = Dense(self.time_steps, name='Acquire_value')(a)  # shape = (batch_size, input_dim, time_steps)

        a_query = Permute((2, 1), name='Query_resume')(a_query)  # shape = (batch_size, time_step, input_dim)
        a_keys = Permute((2, 1), name='Key_resume')(a_keys)  # shape = (batch_size, time_step, input_dim)
        # a_values = Permute((2, 1), name='Value_resume')(a_values)  # shape = (batch_size, time_step, input_dim)

        att = Attention(use_scale=True, name=layer_name)(
            [a_query, a_keys])  # shape = (batch_size, time_step, input_dims) #层内已经自带softmax效果
        return att

    def Costom_attention_2(self, inputs, layer_name='Secondary_attention'):
        """这个模型使用两个注意力机制，两层LSTM"""
        a = Permute((2, 1), name='Transpose_acceleration_2')(inputs)  ## shape = (batch_size, input_dim, time_steps)
        a_query = Dense(self.time_steps, activation='softmax', name='Acquire_query_2')(
            a)  # shape = (batch_size, input_dim, time_steps)
        # 正常的输出应该是batch_size, input_dim, time_steps, 如果不先进行坐标轴变换，就会输出 batch_size, time_steps, time_steps
        # 这里一定首先得进行维度变换，变换后的Dense层才是在每一个序列内部进行q、k、v的计算

        a_keys = Dense(self.time_steps, name='Acquire_key_2')(a)  # shape = (batch_size, input_dim, time_steps)
        # a_values = Dense(self.time_steps, name='Acquire_value_2')(a)  # shape = (batch_size, input_dim, time_steps)

        a_query = Permute((2, 1), name='Query_resume_2')(a_query)  # shape = (batch_size, time_step, input_dim)
        a_keys = Permute((2, 1), name='Key_resume_2')(a_keys)  # shape = (batch_size, time_step, input_dim)
        # a_values = Permute((2, 1), name='Value_resume_2')(a_values)  # shape = (batch_size, time_step, input_dim)

        att = Attention(use_scale=True, name=layer_name)(
            [a_query, a_keys])  # shape = (batch_size, time_step, input_dims)

        return att

    def model_attention_applied_before_lstm(self, inputs):
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
        zeros = inputs[:, -1, 2:3] - inputs[:, -1, 2:3]  # 生成一个在虚拟模型状态形状为None, 1 的零向量，作为温度的补齐
        dV = tf.concat(values=[inputs[:, -1, 2:3], zeros], axis=1)  # 序号为2的列是dV_static_star，因此需要补齐

        V_T = inputs[:, -1, 0:2]  # 序号为0，1的列分别为上一时刻的电压、温度
        V_ori = inputs[:, -1, 0:1] * 2.5
        T_ori = inputs[:, -1, 1:2] * 100
        V_T = tf.concat(values = [V_ori,T_ori] , axis = 1)
        att = self.Costom_attention_1(inputs, layer_name='Primary_attention')
        att = self.Costom_attention_2(att, layer_name='Secondary_attention')

        # attention_mul = Multiply(name = 'Attention_mechanism')([inputs,a_query]) # shape = (batch_size, time_step, input_dim)
        stepping = LSTM(8, return_sequences=True, name='LSTM_layer_1')(att)
        stepping = LSTM(8, return_sequences=False, name='LSTM_layer_2')(stepping)
        outputs = Dense(self.output_dims, activation='sigmoid', name='Conclude')(stepping)
        residual = outputs + V_T + dV

        return residual

    def get_model(self):
        input_data = Input(shape=(self.time_steps, self.input_dims), name='data_input')
        output_data = self.model_attention_applied_before_lstm(input_data)

        model = Model(input_data, output_data)
        # model.compile(optimizer='adam', loss = 'binary_crossentropy',metrics=['accuracy'])
        return model

class Attention_LSTM_no_dV_static():
    """在这个模型中输入的dV不再在最后输出前加和到原始电压之上，只作为简单的输入"""
    def __init__(self, time_steps=64, input_dims=7, output_dims=2):
        self.time_steps = time_steps
        self.input_dims = input_dims  # 目前的模型里面应该是7
        self.output_dims = output_dims  # 目前的模型里面只能是2

    def Costom_attention_1(self, inputs, layer_name='Primary_attention'):
        """这个模型使用两个注意力机制，两层LSTM"""
        a = Permute((2, 1), name='Transpose_acceleration')(inputs)  ## shape = (batch_size, input_dim, time_steps)
        a_query = Dense(self.time_steps, activation='softmax', name='Acquire_query')(
            a)  # shape = (batch_size, input_dim, time_steps)
        # 正常的输出应该是batch_size, input_dim, time_steps, 如果不先进行坐标轴变换，就会输出 batch_size, time_steps, time_steps
        # 这里一定首先得进行维度变换，变换后的Dense层才是在每一个序列内部进行q、k、v的计算

        a_keys = Dense(self.time_steps, name='Acquire_key')(a)  # shape = (batch_size, input_dim, time_steps)

        a_query = Permute((2, 1), name='Query_resume')(a_query)  # shape = (batch_size, time_step, input_dim)
        a_keys = Permute((2, 1), name='Key_resume')(a_keys)  # shape = (batch_size, time_step, input_dim)

        att = Attention(use_scale=True, name=layer_name)(
            [a_query, a_keys])  # shape = (batch_size, time_step, input_dims) #层内已经自带softmax效果
        return att

    def Costom_attention_2(self, inputs, layer_name='Secondary_attention'):
        """这个模型使用两个注意力机制，两层LSTM"""
        a = Permute((2, 1), name='Transpose_acceleration_2')(inputs)  ## shape = (batch_size, input_dim, time_steps)
        a_query = Dense(self.time_steps, activation='softmax', name='Acquire_query_2')(
            a)  # shape = (batch_size, input_dim, time_steps)
        # 正常的输出应该是batch_size, input_dim, time_steps, 如果不先进行坐标轴变换，就会输出 batch_size, time_steps, time_steps
        # 这里一定首先得进行维度变换，变换后的Dense层才是在每一个序列内部进行q、k、v的计算
        a_keys = Dense(self.time_steps, name='Acquire_key_2')(a)  # shape = (batch_size, input_dim, time_steps)

        a_query = Permute((2, 1), name='Query_resume_2')(a_query)  # shape = (batch_size, time_step, input_dim)
        a_keys = Permute((2, 1), name='Key_resume_2')(a_keys)  # shape = (batch_size, time_step, input_dim)

        att = Attention(use_scale=True, name=layer_name)(
            [a_query, a_keys])  # shape = (batch_size, time_step, input_dims)

        return att

    def model_attention_applied_before_lstm(self, inputs):
        V_ori = inputs[:, -1, 0:1] * 2.5
        T_ori = inputs[:, -1, 1:2] * 100
        V_T = tf.concat(values = [V_ori,T_ori] , axis = 1)
        att = self.Costom_attention_1(inputs, layer_name='Primary_attention')
        att = self.Costom_attention_2(att, layer_name='Secondary_attention')

        stepping = LSTM(8, return_sequences=True, name='LSTM_layer_1')(att)
        stepping = LSTM(8, return_sequences=False, name='LSTM_layer_2')(stepping)
        outputs = Dense(self.output_dims, activation='sigmoid', name='Conclude')(stepping)
        residual = outputs + V_T

        return residual

    def get_model(self):
        input_data = Input(shape=(self.time_steps, self.input_dims), name='data_input')
        output_data = self.model_attention_applied_before_lstm(input_data)
        model = Model(input_data, output_data)
        return model


class Only_LSTM():
    """在这个模型中输入的dV不再在最后输出前加和到原始电压之上，只作为简单的输入"""
    def __init__(self, time_steps=64, input_dims=7, output_dims=2):
        self.time_steps = time_steps
        self.input_dims = input_dims  # 目前的模型里面应该是7
        self.output_dims = output_dims  # 目前的模型里面只能是2

    def Costom_attention_1(self, inputs, layer_name='Primary_attention'):
        """这个模型使用两个注意力机制，两层LSTM"""
        a = Permute((2, 1), name='Transpose_acceleration')(inputs)  ## shape = (batch_size, input_dim, time_steps)
        a_query = Dense(self.time_steps, activation='softmax', name='Acquire_query')(
            a)  # shape = (batch_size, input_dim, time_steps)
        # 正常的输出应该是batch_size, input_dim, time_steps, 如果不先进行坐标轴变换，就会输出 batch_size, time_steps, time_steps
        # 这里一定首先得进行维度变换，变换后的Dense层才是在每一个序列内部进行q、k、v的计算

        a_keys = Dense(self.time_steps, name='Acquire_key')(a)  # shape = (batch_size, input_dim, time_steps)

        a_query = Permute((2, 1), name='Query_resume')(a_query)  # shape = (batch_size, time_step, input_dim)
        a_keys = Permute((2, 1), name='Key_resume')(a_keys)  # shape = (batch_size, time_step, input_dim)

        att = Attention(use_scale=True, name=layer_name)(
            [a_query, a_keys])  # shape = (batch_size, time_step, input_dims) #层内已经自带softmax效果
        return att

    def Costom_attention_2(self, inputs, layer_name='Secondary_attention'):
        """这个模型使用两个注意力机制，两层LSTM"""
        a = Permute((2, 1), name='Transpose_acceleration_2')(inputs)  ## shape = (batch_size, input_dim, time_steps)
        a_query = Dense(self.time_steps, activation='softmax', name='Acquire_query_2')(
            a)  # shape = (batch_size, input_dim, time_steps)
        # 正常的输出应该是batch_size, input_dim, time_steps, 如果不先进行坐标轴变换，就会输出 batch_size, time_steps, time_steps
        # 这里一定首先得进行维度变换，变换后的Dense层才是在每一个序列内部进行q、k、v的计算
        a_keys = Dense(self.time_steps, name='Acquire_key_2')(a)  # shape = (batch_size, input_dim, time_steps)

        a_query = Permute((2, 1), name='Query_resume_2')(a_query)  # shape = (batch_size, time_step, input_dim)
        a_keys = Permute((2, 1), name='Key_resume_2')(a_keys)  # shape = (batch_size, time_step, input_dim)

        att = Attention(use_scale=True, name=layer_name)(
            [a_query, a_keys])  # shape = (batch_size, time_step, input_dims)

        return att

    def model_attention_applied_before_lstm(self, inputs):
        V_ori = inputs[:, -1, 0:1] * 2.5
        T_ori = inputs[:, -1, 1:2] * 100
        V_T = tf.concat(values = [V_ori,T_ori] , axis = 1)
        att = self.Costom_attention_1(inputs, layer_name='Primary_attention')
        # att = self.Costom_attention_2(att, layer_name='Secondary_attention')

        stepping = LSTM(32, return_sequences=True, name='LSTM_layer_1')(inputs)
        stepping = LSTM(32, return_sequences=False, name='LSTM_layer_2')(stepping)
        outputs = Dense(self.output_dims, activation='sigmoid', name='Conclude')(stepping)
        residual = outputs + V_T

        return residual

    def get_model(self):
        input_data = Input(shape=(self.time_steps, self.input_dims), name='data_input')
        output_data = self.model_attention_applied_before_lstm(input_data)
        model = Model(input_data, output_data)
        return model



# model = Attention_LSTM()
# model = model.get_model()
# print(model(x).shape)
# model.summary()
#
# from keras.utils import plot_model
# plot_model(model,to_file='Cache/model2.png',dpi = 300)
