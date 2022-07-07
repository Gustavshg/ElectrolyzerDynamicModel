import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Permute, Input, Multiply, Add, Attention



class Attention_LSTM_no_residual():
    """模型中不只没有dV_static_star， 同时还将模型中最后的原始数据加和在结果中进行了去除，即没有残差操作"""
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
        residual = outputs + V_T - V_T

        return residual

    def get_model(self):
        input_data = Input(shape=(self.time_steps, self.input_dims), name='data_input')
        output_data = self.model_attention_applied_before_lstm(input_data)
        model = Model(input_data, output_data)
        return model

class Attention_LSTM_spaced():
    """模型中不只没有dV_static_star， 同时还将模型中最后的原始数据加和在结果中进行了去除，即没有残差操作"""
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

    def Costom_attention_3(self, inputs, layer_name='Secondary_attention'):
        """这个模型使用两个注意力机制，两层LSTM"""
        a = Permute((2, 1), name='Transpose_acceleration_3')(inputs)  ## shape = (batch_size, input_dim, time_steps)
        a_query = Dense(self.time_steps, activation='softmax', name='Acquire_query_3')(
            a)  # shape = (batch_size, input_dim, time_steps)
        # 正常的输出应该是batch_size, input_dim, time_steps, 如果不先进行坐标轴变换，就会输出 batch_size, time_steps, time_steps
        # 这里一定首先得进行维度变换，变换后的Dense层才是在每一个序列内部进行q、k、v的计算
        a_keys = Dense(self.time_steps, name='Acquire_key_3')(a)  # shape = (batch_size, input_dim, time_steps)

        a_query = Permute((2, 1), name='Query_resume_3')(a_query)  # shape = (batch_size, time_step, input_dim)
        a_keys = Permute((2, 1), name='Key_resume_3')(a_keys)  # shape = (batch_size, time_step, input_dim)

        att = Attention(use_scale=True, name=layer_name)(
            [a_query, a_keys])  # shape = (batch_size, time_step, input_dims)

        return att

    def model_attention_applied_before_lstm(self, inputs):
        V_ori = inputs[:, -1, 0:1] * 2.5
        T_ori = inputs[:, -1, 1:2] * 100
        V_T = tf.concat(values = [V_ori,T_ori] , axis = 1)
        lstm1 = LSTM(self.input_dims , return_sequences=True, name='Data_feature')(inputs)
        att = self.Costom_attention_1(lstm1, layer_name='Primary_attention')
        stepping1 = LSTM(self.input_dims, return_sequences=True, name='LSTM_layer_1')(att)

        att = self.Costom_attention_2(lstm1, layer_name='Secondary_attention')
        stepping2= LSTM(self.input_dims, return_sequences=False, name='LSTM_layer_2')(att)
        stepping = stepping1 + inputs
        att = self.Costom_attention_3(stepping, layer_name='Conclusion_attention')
        stepping = LSTM(self.input_dims, return_sequences=False, name='LSTM_layer_3')(att)
        stepping = stepping2 + stepping
        outputs = Dense(self.output_dims, activation='sigmoid', name='Conclude')(stepping)
        residual = outputs

        return residual

    def get_model(self):
        input_data = Input(shape=(self.time_steps, self.input_dims), name='data_input')
        output_data = self.model_attention_applied_before_lstm(input_data)
        model = Model(input_data, output_data)
        return model
