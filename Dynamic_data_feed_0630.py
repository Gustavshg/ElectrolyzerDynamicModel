import pandas
import numpy as np
import os
import Smoothen as sm

def to_np_and_expand(data):
    """这里是将原始列表进行扩展，变成符合输入需求的一维向量，后续仍需要进行数组的连接才能输入模型"""
    res = np.array(data)
    res = np.expand_dims(res, axis=1)
    return res




class dynamic_data():
    """这个里面是比较简单的模型数据的尝试，就是简单的五个数据内容"""

    def __init__(self, batch_size=50, source_folder='Dynamic model data-20s/Data 0614'):
        self.batch_size = batch_size
        self.source_folder = source_folder
        self.file_list = os.listdir(self.source_folder)
        self.file_list.sort()
        self.total_length = 0
        for file in self.file_list:
            file = os.path.join(self.source_folder, file)
            df = pandas.read_csv(file)
            self.total_length += len(df)

        self.file_num = 5

        self.ALL_columns = ['时间', '电解电压', '电解电流',
                            '产氢量', '产氢累计量', '碱液流量', '碱温', '系统压力  ', '氧槽温', '氢槽温', '氧侧液位', '氢侧液位',
                            '氧中氢', '氢中氧', '脱氧上温', '脱氧下温', 'B塔上温', 'B塔下温', 'C塔上温', 'C塔下温', 'A塔上温',
                            'A塔下温', '露点', '微氧量', '出罐压力', '进罐温度', '进罐压力', 'V', 'I',
                            'Current density', 'Pressure', 'Tlye', 'TH2', 'TO2', 'Tout', 'LyeFlow',
                            'LyeFlow_Polar', 'dI', 'dj', 'dV', 'dTout', 'HTO', 'OTH', 'polar_lh',
                            'polar_wtt', 'polar_shg', 'polar_nn', 'T_out_star', 'V_static_star',
                            'dV_static_star', 'dV_dynamic_star', 'dTout_WL', 'V_star']

    def get_input_cols(self, df):
        V_star = to_np_and_expand(df['V_star']) / 2.2  # 上一个采样时刻的电压
        T_out_star = to_np_and_expand(df['T_out_star']) / 85.  # 上一个采样时刻的温度
        dV_static_star = to_np_and_expand(df['dV_static_star'])
        Current_density = to_np_and_expand(df['Current density']) / 4000
        T_in = to_np_and_expand(df['Tlye']) / 80
        Lye_flow = to_np_and_expand(df['LyeFlow']) / 2
        Amb_temp = to_np_and_expand(df['AmbT']) / 50
        dJ = to_np_and_expand(df['dj']) / 100
        # inputs = np.concatenate((V_star, T_out_star, dV_static_star, Current_density, T_in, Lye_flow,Amb_temp, dJ), axis=1)
        inputs = np.concatenate((V_star, T_out_star, Current_density, T_in, Lye_flow, Amb_temp, dJ), axis=1)
        V = to_np_and_expand(df['V'])/2.2
        T_out = to_np_and_expand(df['Tout'])/85.

        return inputs, V, T_out

    def get_batch(self, num_step=60):
        """已经修改好了，可以正常使用"""
        X = []
        Y_V = []
        Y_T = []
        self.batch_file_list = np.random.randint(low=0, high=len(self.file_list),
                                                 size=self.file_num)  # 随机抽取5个文件，每个文件里抽取10个sequence，每次抽取批次都需要重新生成随即列表
        for i in range(len(self.batch_file_list)):
            file = self.batch_file_list[i]
            file = self.file_list[file]
            file = os.path.join(self.source_folder, file)
            df = pandas.read_csv(file)
            inputs, V, T_out = self.get_input_cols(df)

            random_locs = np.random.randint(low=0, high=len(df) - num_step,
                                            size=self.batch_size // self.file_num)  # 这里最后到length-num_step为止
            for idk in random_locs:
                X.append(inputs[idk:idk + num_step, :])
                Y_V.append(V[idk + num_step - 1])
                Y_T.append(T_out[idk + num_step - 1])
        X = np.array(X)
        Y_V = np.array(Y_V)
        Y_T = np.array(Y_T)
        Y = np.concatenate((Y_V, Y_T), axis=1)
        return X, Y

    def get_easy_batch(self, batch_size=50, num_step=60):
        """这里暂时是只提取固定日期的数据"""
        df = pandas.read_csv('Dynamic model data-20s/Data 0614/TJ-20211129.csv')
        inputs, V, T_out = self.get_input_cols(df)

        random_locs = np.random.randint(low=0, high=len(df) - num_step,
                                        size=self.batch_size // self.file_num)  # 这里最后到length-num_step为止
        X = []
        Y_V = []
        Y_T = []
        for idk in random_locs:
            X.append(inputs[idk:idk + num_step, :])
            Y_V.append(V[idk + num_step - 1])
            Y_T.append(T_out[idk + num_step - 1])
        X = np.array(X)
        Y_V = np.array(Y_V)
        Y_T = np.array(Y_T)
        Y = np.concatenate((Y_V, Y_T), axis=1)
        return X, Y

    def get_redemo_data(self, num_step=60, date='1129'):
        """这里暂时是只提取固定日期的数据，供模型测试时复现数据，以直观展示结果的进步过程"""
        file = 'Dynamic model data-20s/Data 0614/TJ-2021' + date + '.csv'
        df = pandas.read_csv(file)

        inputs, V, T_out = self.get_input_cols(df)
        """在给出对比结果的时候，可以忽略前六十个点，只给num_step之后的结果，输入输出都是这样"""
        X = []
        Y_V = []
        Y_T = []
        for idk in range(0, len(df) - num_step):
            X.append(inputs[idk:idk + num_step, :])
            Y_V.append(V[idk + num_step - 1])
            Y_T.append(T_out[idk + num_step - 1])
        return X, Y_V, Y_T

    def print_ori_v_t(self, num_step=60, date='1129'):
        """这里返回我们使用日期的目标数据，看一下模型最后退回数据的情况，复现整个电压曲线看一下结果"""
        file = 'Dynamic model data-20s/Data 0614/TJ-2021' + date + '.csv'
        df = pandas.read_csv(file)
        dV = list(df['dV'])
        dT_out = list(df['dTout'])
        dV_static_star = list(df['dV_static_star'])

        dV = dV[num_step:]
        dT_out = dT_out[num_step:]

        return dV, dT_out

    def get_model_fit_data(self, num_step=60):
        """这个函数可以一次性输出所有的数据，让模型开始自己进行split并且训练"""
        X = []
        Y_V = []
        Y_T = []
        for file in self.file_list:
            file = os.path.join(self.source_folder, file)
            df = pandas.read_csv(file)
            inputs, V, T_out = self.get_input_cols(df)
            for idx in range(0, len(df) - num_step):
                X.append(inputs[idx:idx + num_step, :])
                Y_V.append(V[idx + num_step - 1])
                Y_T.append(T_out[idx + num_step - 1])
        X = np.array(X)
        Y_V = np.array(Y_V)
        Y_T = np.array(Y_T)
        Y = np.concatenate((Y_V, Y_T), axis=1)
        return X, Y






"""

这里是生成开机过程所需要的数据和函数

"""

def Current_density_start_up(j0=0., j1=3500.):
    """这一部分可以得到一个数组，计算好了开机过程中的电流密度和变化，电流密度变化的时间和电流密度大小相关，后面可以直接接后面的电流密度设定值"""
    time_span = int(abs(j1 - j0) // 700. + 2)
    k = 1 / (1 / ((time_span + 1) / 10) + 0.3)
    time_line = np.arange(time_span)
    cd = (j1 - j0) * (1 - np.exp(-k * time_line)) + j0
    return cd


def Lye_heat_up(lye_time=300, k=0.03, high=61, low=21):
    """这个函数是一个碱液升温的s曲线函数，整个升温过程长度为lye_time，最终温度为high+ambT"""
    time_line = np.arange(lye_time)
    Tlye = (high - low) / (1 + np.exp(-k * (time_line - lye_time / 2))) + low
    return Tlye


def Lye_cool_down(timespan=600, high=61, low=21, k=0.007):
    """这个函数是一个碱液降温的s曲线函数，实际上为s曲线的一般，整个时间长度为time_span"""
    time_line = np.arange(timespan)
    Tlye = (high - low) * np.exp(-k * time_line) + low
    return Tlye

class Artificial_data():
    def __init__(self, num_step=32):
        """所有有关num_step的数据处理都放在最后输出数据前再进行"""
        """每个时间点的长度是20s"""
        self.num_step = num_step

    def num_step_process(self, inputs):
        """所有的原始数据生成之后，可以利用这个函数将其输出成为可以用在模型输入里面的内容"""
        """这个函数可能暂时没用，因为我们没法得到全部的初始数据"""
        X = []
        for idx in range(0, len(inputs) - self.num_step):
            X.append(inputs[idx:idx + self.num_step, :])
        return X

    def to_input_row(self, data):
        """这里是把输入的一个向量变成一个复合模型输入规则的一个立体向量，以方便给出输入矩阵"""
        return np.array(data).reshape((self.num_step, 1))

    def start_up(self, current_density_setpoint=3500, v0=0.00027, t0=22.034, lye_time=300, lye_flow=1.3, t_lye_0=22.181,
                 t_lye=61.5, wait_time=100, close_time=2500, total_time=3500):
        """这一部分需要根据开机前时间、开机时刻、稳定电流、稳定碱液温度等给出一个开机的数据"""
        """输入变量的含义：
            current_density_setpoint 开机后额定的电解槽工作电流密度
            v0 开始计算时的电解槽电压，在这个函数里就是关机静置状态的电压
            t0 同上，静置状态的电解槽出口温度
            lye_time 电解槽从开始工作到碱液温度上升到60摄氏度需要的时间点长度，每个时间点为20s
            lye_flow 电解槽工作的碱液流量
            t_lye_0 静置状态下的电解槽碱液入口温度
            t_lye 正常工作时碱液稳定地温度
            wait_time开机前的静置时间
            close_time电流密度开始下降的时间位置
            total_time 整个序列的时间点长度，每个时间点为20s
        """
        """V_star, T_out_star,  Current_density, T_in, Lye_flow, Amb_temp, dJ"""
        """ 开机前 """
        V_star = [v0] * self.num_step  # 初始电压，关机状态下，这个状态会一直持续到初始关机状态的结束
        T_out_star = [t0] * self.num_step  # 初始温度，关机状态下，这个状态会一直持续到初始关机状态的结束
        Current_density = [0.] * wait_time  # 初始电流密度，关机状态下，这个状态会一直持续到初始关机状态的结束
        T_in = [t_lye_0] * wait_time
        Lye_flow = [0.] * wait_time
        Amb_temp = [t_lye_0] * wait_time  # 初始环境温度设置为碱液流量的初始温度
        # dJ 可以在最后结束生成数列之后，再用完整的电流密度数列来产生

        """ 开机过程及后续过程 """
        # 电流密度数据
        cd_start_up = Current_density_start_up(j0=0., j1=3500.)
        for cur in cd_start_up:
            Current_density.append(cur)
        while len(Current_density) < close_time:
            Current_density.append(current_density_setpoint)
        cd_close_down = Current_density_start_up(j0=3500., j1=0.)
        for cur in cd_close_down:
            Current_density.append(cur)
        while len(Current_density) < total_time:
            Current_density.append(0.)
        # 碱液入口温度
        tl_start_up = Lye_heat_up(lye_time=lye_time, high=t_lye, low=t_lye_0)
        for cur in tl_start_up:
            T_in.append(cur)
        while len(T_in) < close_time:
            T_in.append(t_lye)
        tl_close_down = Lye_cool_down(high=t_lye, low=t_lye_0)
        for cur in tl_close_down:
            T_in.append(cur)
        while len(T_in) < total_time:
            T_in.append(t_lye_0)
        # 碱液流量
        while len(Lye_flow) < close_time:
            Lye_flow.append(lye_flow)
        while len(Lye_flow) < total_time:
            Lye_flow.append(0.)
        # 环境温度
        while len(Amb_temp) < total_time:
            Amb_temp.append(t_lye_0)

        # 电流密度微分数据
        dJ = sm.diff(Current_density)
        return V_star, T_out_star, Current_density, T_in, Lye_flow, Amb_temp, dJ



def step(model, idx, num_step, V_star, T_out_star, Current_density, T_in, Lye_flow, Amb_temp, dJ):
    """模型的步进函数，可以依照一整个输入序列，根据不同位置的安排，每次计算下一个点"""
    # 首先抽取当前计算所需的部分
    V_star_cur = V_star[idx:idx + num_step]
    T_out_star_cur = T_out_star[idx:idx + num_step]
    Current_density_cur = Current_density[idx:idx + num_step]
    T_in_cur = T_in[idx:idx + num_step]
    Lye_flow_cur = Lye_flow[idx:idx + num_step]
    Amb_temp_cur = Amb_temp[idx:idx + num_step]
    dJ_cur = dJ[idx:idx + num_step]
    # 对当前计算部分数列进行格式调整
    V_star_cur = to_np_and_expand(V_star_cur)
    T_out_star_cur = to_np_and_expand(T_out_star_cur)
    Current_density_cur = to_np_and_expand(Current_density_cur)
    T_in_cur = to_np_and_expand(T_in_cur)
    Lye_flow_cur = to_np_and_expand(Lye_flow_cur)
    Amb_temp_cur = to_np_and_expand(Amb_temp_cur)
    dJ_cur = to_np_and_expand(dJ_cur)
    # 对输入进行归一化，以符合模型需求
    inputs = np.concatenate((V_star_cur / 2.2, T_out_star_cur / 85., Current_density_cur / 4000., T_in_cur / 80.,
                             Lye_flow_cur / 2., Amb_temp_cur / 50., dJ_cur / 100.), axis=1)
    inputs = np.expand_dims(inputs, axis=0)
    outputs = model(inputs)
    # 对计算结果进行提取，得到最终的结果
    V_next = outputs.numpy()[0][0]
    T_next = outputs.numpy()[0][1]
    return V_next, T_next

def step_no_dJ(model, idx, num_step, V_star, T_out_star, Current_density, T_in, Lye_flow, Amb_temp, dJ):
    """模型的步进函数，可以依照一整个输入序列，根据不同位置的安排，每次计算下一个点"""
    """在有的模型中，不需要输入电解槽的电流密度微分，所以这里就是为了适应这种模型"""
    # 首先抽取当前计算所需的部分
    V_star_cur = V_star[idx:idx + num_step]
    T_out_star_cur = T_out_star[idx:idx + num_step]
    Current_density_cur = Current_density[idx:idx + num_step]
    T_in_cur = T_in[idx:idx + num_step]
    Lye_flow_cur = Lye_flow[idx:idx + num_step]
    Amb_temp_cur = Amb_temp[idx:idx + num_step]
    dJ_cur = dJ[idx:idx + num_step]
    # 对当前计算部分数列进行格式调整
    V_star_cur = to_np_and_expand(V_star_cur)
    T_out_star_cur = to_np_and_expand(T_out_star_cur)
    Current_density_cur = to_np_and_expand(Current_density_cur)
    T_in_cur = to_np_and_expand(T_in_cur)
    Lye_flow_cur = to_np_and_expand(Lye_flow_cur)
    Amb_temp_cur = to_np_and_expand(Amb_temp_cur)
    dJ_cur = to_np_and_expand(dJ_cur)
    # 对输入进行归一化，以符合模型需求
    inputs = np.concatenate((V_star_cur / 2.2, T_out_star_cur / 85., Current_density_cur / 4000., T_in_cur / 100.,
                             Lye_flow_cur / 2., Amb_temp_cur / 100.), axis=1)
    inputs = np.expand_dims(inputs, axis=0)
    outputs = model(inputs)
    # 对计算结果进行提取，得到最终的结果
    V_next = outputs.numpy()[0][0]
    T_next = outputs.numpy()[0][1]
    return V_next, T_next

# import keras
# import matplotlib.pyplot as plt
# model = keras.models.load_model(
#     'Neural Networks/Dynamic model/Version 0623/trial attention lstm double simplified 32 step no compile and fit 1.5.ckpt')
# model.summary()
# num_step = 32
# data = Artificial_data(num_step=num_step)
# # 这里可以自己创建一个开机过程的基础数据
#
# V_star, T_out_star, Current_density, T_in, Lye_flow, Amb_temp, dJ = data.start_up(current_density_setpoint=3500,
#                                                                                   v0=0.00027, t0=22.034, lye_time=300,
#                                                                                   lye_flow=1.3, t_lye_0=22.181,
#                                                                                   t_lye=61.5, wait_time=100,
#                                                                                   close_time=2500, total_time=3500)
# for idx in range(len(Current_density) - num_step):
#     V_next,T_next = step(model,idx,num_step,V_star, T_out_star, Current_density, T_in, Lye_flow, Amb_temp, dJ )
#     V_star.append(V_next)
#     T_out_star.append(T_next)
#     print(idx,V_next,T_next)
# plt.figure()
# plt.plot(V_star)
# plt.title('V predicted')
# plt.figure()
# plt.plot(T_out_star)
# plt.title('T predicted')
# plt.show()

# 这里可以读取一天的数据作为输入数据
# df = pandas.read_csv('Dynamic model data-20s/Data 0614/TJ-20210924.csv')
# V_star = list(df['V_star'])
# T_out_star = list(df['T_out_star'])
# Current_density = list(df['Current density'])
# T_in = list(df['Tlye'])
# Lye_flow = list(df['LyeFlow'])
# Amb_temp = list(df['AmbT'])
# dJ = list(df['dj'])
# print(len(V_star))
# V_pred = []
# T_pred = []
#
# for idx in range(len(Current_density) - num_step):
#     V_next,T_next = step(model,idx,num_step,V_star, T_out_star, Current_density, T_in, Lye_flow, Amb_temp, dJ )
#     V_pred.append(V_next)
#     T_pred.append(T_next)
#     print(idx,V_next,T_next)
# plt.figure()
# plt.plot(V_pred)
# plt.title('V predicted')
# plt.figure()
# plt.plot(T_pred)
# plt.title('T predicted')
# plt.show()
