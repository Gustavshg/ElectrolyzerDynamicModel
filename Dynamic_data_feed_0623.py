import pandas
import numpy as np
import os

def to_np_and_expand(data):
    res = np.array(data)
    res = np.expand_dims(res,axis=1)
    return res




class dynamic_data():
    """这个里面是比较简单的模型数据的尝试，就是简单的五个数据内容"""
    def __init__(self,batch_size = 50,source_folder = 'Dynamic model data-20s/Data 0614'):
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

    def get_input_cols(self,df):
        V_star = to_np_and_expand(df['V_star'])/2.5  # 上一个采样时刻的电压
        T_out_star = to_np_and_expand(df['T_out_star'])/100# 上一个采样时刻的温度
        dV_static_star = to_np_and_expand(df['dV_static_star'])
        Current_density = to_np_and_expand(df['Current density'])/4000
        T_in = to_np_and_expand(df['Tlye'])/100
        Lye_flow = to_np_and_expand(df['LyeFlow'])/2
        Amb_temp = to_np_and_expand(df['AmbT'])/100
        dJ = to_np_and_expand(df['dj'])/100
        # inputs = np.concatenate((V_star, T_out_star, dV_static_star, Current_density, T_in, Lye_flow,Amb_temp, dJ), axis=1)
        inputs = np.concatenate((V_star, T_out_star,  Current_density, T_in, Lye_flow, Amb_temp, dJ), axis=1)
        V = to_np_and_expand(df['V'])
        T_out =to_np_and_expand(df['Tout'])

        return inputs,V,T_out

    def get_batch(self,num_step = 60):
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
            inputs,V,T_out = self.get_input_cols(df)

            random_locs = np.random.randint(low=0, high=len(df) - num_step,
                                            size=self.batch_size // self.file_num)  # 这里最后到length-num_step为止
            for idk in random_locs:
                X.append(inputs[idk:idk + num_step, :])
                Y_V.append(V[idk + num_step - 1])
                Y_T.append(T_out[idk + num_step - 1])
        X = np.array(X)
        Y_V = np.array(Y_V)
        Y_T = np.array(Y_T)
        Y = np.concatenate((Y_V,Y_T),axis = 1)
        return X, Y



    def get_easy_batch(self,batch_size = 50,num_step = 60):
        """这里暂时是只提取固定日期的数据"""
        df = pandas.read_csv('Dynamic model data-20s/Data 0614/TJ-20211129.csv')
        inputs, V,T_out = self.get_input_cols(df)

        random_locs = np.random.randint(low=0, high=len(df)-num_step, size=self.batch_size // self.file_num)  # 这里最后到length-num_step为止
        X = []
        Y_V = []
        Y_T = []
        for idk in random_locs:
            X.append(inputs[idk:idk+num_step,:])
            Y_V.append(V[idk + num_step-1])
            Y_T.append(T_out[idk+num_step-1])
        X = np.array(X)
        Y_V = np.array(Y_V)
        Y_T = np.array(Y_T)
        Y = np.concatenate((Y_V,Y_T),axis = 1)
        return X,Y

    def get_redemo_data(self,num_step = 60,date = '1129'):
        """这里暂时是只提取固定日期的数据，供模型测试时复现数据，以直观展示结果的进步过程"""
        file = 'Dynamic model data-20s/Data 0614/TJ-2021' + date + '.csv'
        df = pandas.read_csv(file)

        inputs, V,T_out = self.get_input_cols(df)
        """在给出对比结果的时候，可以忽略前六十个点，只给num_step之后的结果，输入输出都是这样"""
        X = []
        Y_V = []
        Y_T = []
        for idk in range(0,len(df) - num_step):
            X.append(inputs[idk:idk + num_step, :])
            Y_V.append(V[idk + num_step - 1])
            Y_T.append(T_out[idk + num_step - 1])
        return X,Y_V,Y_T

    def print_ori_v_t(self,num_step = 60,date = '1129'):
        """这里返回我们使用日期的目标数据，看一下模型最后退回数据的情况，复现整个电压曲线看一下结果"""
        file = 'Dynamic model data-20s/Data 0614/TJ-2021' + date + '.csv'
        df = pandas.read_csv(file)
        dV = list(df['dV'])
        dT_out = list(df['dTout'])
        dV_static_star = list(df['dV_static_star'])


        dV = dV[num_step:]
        dT_out = dT_out[num_step:]

        return dV,dT_out

    def get_model_fit_data(self,num_step = 60):
        """这个函数可以一次性输出所有的数据，让模型开始自己进行split并且训练"""
        X = []
        Y_V = []
        Y_T = []
        for file in self.file_list:
            file = os.path.join(self.source_folder, file)
            df = pandas.read_csv(file)
            inputs,V,T_out = self.get_input_cols(df)
            for idx in range(0, len(df) - num_step):
                X.append(inputs[idx:idx + num_step, :])
                Y_V.append(V[idx + num_step - 1])
                Y_T.append(T_out[idx + num_step - 1])
        X = np.array(X)
        Y_V = np.array(Y_V)
        Y_T = np.array(Y_T)
        Y = np.concatenate((Y_V,Y_T),axis = 1)
        return X,Y

class Artificial_data():
    def __init__(self,num_step = 32):
        """所有有关num_step的数据处理都放在最后输出数据前再进行"""
        """每个时间点的长度是20s"""
        import keras
        self.num_step = num_step
        self.model = keras.models.load_model( 'Neural Networks/Dynamic model/Version 0623/trial attention lstm double simplified 32 step no compile and fit 1.7.ckpt')

    def num_step_process(self,inputs):
        """所有的原始数据生成之后，可以利用这个函数将其输出成为可以用在模型输入里面的内容"""
        """这个函数可能暂时没用，因为我们没法得到全部的初始数据"""
        X = []
        for idx in range(0, len(inputs) - self.num_step):
            X.append(inputs[idx:idx + self.num_step, :])
        return X

    def to_input_row(self,data):
        """这里是把输入的一个向量变成一个复合模型输入规则的一个立体向量，以方便给出输入矩阵"""
        return np.array(data).reshape((self.num_step,1))

    def start_up(self,wait_time=100,current_density=3500,v0=0.00027,t0 = 22.034,lye_time=180,lye_flow = 1.3,t_lye_0=22.181,t_lye=61.5,total_time=1500):
        """这一部分需要根据开机前时间、开机时刻、稳定电流、稳定碱液温度等给出一个开机的数据"""
        """V_star, T_out_star,  Current_density, T_in, Lye_flow, Amb_temp, dJ"""
        V_star = [v0] * self.num_step  # 初始电压，关机状态下
        T_out_star = [t0] * self.num_step  # 初始温度，关机状态下
        Current_density = [0] * self.num_step  # 初始电流密度，关机状态下

    def get_model_fit_data(self,num_step = 32):
        """这个函数可以一次性输出所有的数据，让模型开始自己进行split并且训练"""
        X = []
        Y_V = []
        Y_T = []
        for file in self.file_list:
            file = os.path.join(self.source_folder, file)
            df = pandas.read_csv(file)
            inputs,V,T_out = self.get_input_cols(df)
            for idx in range(0, len(df) - num_step):
                X.append(inputs[idx:idx + num_step, :])
                Y_V.append(V[idx + num_step - 1])
                Y_T.append(T_out[idx + num_step - 1])
        X = np.array(X)
        Y_V = np.array(Y_V)
        Y_T = np.array(Y_T)
        Y = np.concatenate((Y_V,Y_T),axis = 1)
        return X,Y


        print(V_star.shape)
data = Artificial_data()
data.start_up()
