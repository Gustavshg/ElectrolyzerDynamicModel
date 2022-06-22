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
        self.batch_file_list = np.random.randint(low=0, high=len(self.file_list), size=self.file_num)  # 随机抽取5个文件，每个文件里抽取10个sequence
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
        Current_density = to_np_and_expand(df['Current density'])/4000
        T_in = to_np_and_expand(df['Tlye'])/100
        Lye_flow = to_np_and_expand(df['LyeFlow'])/2
        dV_static_star = to_np_and_expand(df['dV_static_star'])
        inputs = np.concatenate((V_star, T_out_star, Current_density, T_in, Lye_flow, dV_static_star), axis=1)

        dV_dynamic_star = to_np_and_expand(df['dV_dynamic_star'])
        dT_out_WL = to_np_and_expand(df['dTout_WL'])

        return inputs,dV_dynamic_star,dT_out_WL

    def get_batch(self,num_step = 60):
        """暂时这个还没法使用"""
        X = []
        Y_V = []
        Y_T = []
        for i in range(len(self.batch_file_list)):
            file = self.batch_file_list[i]
            file = self.file_list[file]
            file = os.path.join(self.source_folder, file)
            df = pandas.read_csv(file)
            inputs, dV_dynamic_star, dT_out_WL = self.get_input_cols(df)

            random_locs = np.random.randint(low=0, high=len(df) - num_step,
                                            size=self.batch_size // self.file_num)  # 这里最后到length-num_step为止
            for idk in random_locs:
                X.append(inputs[idk:idk + num_step, :])
                Y_V.append(dV_dynamic_star[idk + num_step - 1])
                Y_T.append(dT_out_WL[idk + num_step - 1])
        X = np.array(X)
        Y_V = np.array(Y_V)
        Y_T = np.array(Y_T)
        Y = np.concatenate((Y_V,Y_T),axis = 1)
        return X, Y
        #     T_out = df['Tout']
        #     Current_density = df['Current density']
        #     T_in = df['Tlye']
        #     Lye_flow = df['LyeFlow']
        #     V_static = df['V_static']
        #     V_dynamic = df['V_dynamic']
        #     random_locs = np.random.randint(low=num_step + 1, high=len(df),
        #                                     size=self.batch_size // self.file_num)  # 这里需要从51开始，因为不然这样就没办法抽取上一个时刻的出口温度
        #     for j in random_locs:
        #         cur_T_out = T_in[j - num_step - 1:j - 1]  # 出口温度因为是需要预测的，所以需要上一时刻出口温度，因为当前采样时刻的出口温度不知道
        #         cur_Current_density = Current_density[j - num_step:j]
        #         cur_T_in = T_in[j - num_step:j]
        #         cur_Lye_flow = Lye_flow[j - num_step:j]
        #         cur_V_static = V_static[j - num_step:j]
        #         # cur_V_dynamic = V_dynamic[j - num_step:j]  # 这个是拟合的指标
        #         cur_V_dynamic = V_dynamic[j]  # 这个是拟合的指标，只选取当前时刻作为结果，而不是整个seq
        #         X.append([cur_T_out, cur_T_in, cur_Current_density, cur_Lye_flow, cur_V_static])
        #         Y.append(cur_V_dynamic)
        # X = np.array(X).transpose([0, 2, 1])  # 最后会是[batch_size,num_step,features]的格式，这里也可以使用tf.keras.layers.permute来替代
        # # Y = np.array(Y).reshape(batch_size, num_step, 1)
        # Y = np.array(Y).reshape(self.batch_size,  1)
        # return X,Y


    def get_easy_batch(self,batch_size = 50,num_step = 60):
        """这里暂时是只提取固定日期的数据"""
        df = pandas.read_csv('Dynamic model data-20s/Data 0614/TJ-20211129.csv')
        inputs, dV_dynamic_star, dT_out_WL = self.get_input_cols(df)

        random_locs = np.random.randint(low=0, high=len(df)-num_step, size=self.batch_size // self.file_num)  # 这里最后到length-num_step为止
        X = []
        Y_V = []
        Y_T = []
        for idk in random_locs:
            X.append(inputs[idk:idk+num_step,:])
            Y_V.append(dV_dynamic_star[idk + num_step-1])
            Y_T.append(dT_out_WL[idk+num_step-1])
        X = np.array(X)
        Y_V = np.array(Y_V)
        Y_T = np.array(Y_T)
        Y = np.concatenate((Y_V,Y_T),axis = 1)
        return X,Y

    def get_redemo_data(self,num_step = 60,date = '1129'):
        """这里暂时是只提取固定日期的数据，供模型测试时复现数据，以直观展示结果的进步过程"""
        file = 'Dynamic model data-20s/Data 0614/TJ-2021' + date + '.csv'
        df = pandas.read_csv(file)

        inputs, dV_dynamic_star, dT_out_WL = self.get_input_cols(df)
        """在给出对比结果的时候，可以忽略前六十个点，只给num_step之后的结果，输入输出都是这样"""
        X = []
        Y_V = []
        Y_T = []
        for idk in range(0,len(df) - num_step):
            X.append(inputs[idk:idk + num_step, :])
            Y_V.append(dV_dynamic_star[idk + num_step - 1])
            Y_T.append(dT_out_WL[idk + num_step - 1])
        return X,Y_V,Y_T

    def print_ori_v_t(self,num_step = 60,date = '1129'):
        """这里返回我们使用日期的目标数据，看一下模型最后退回数据的情况，复现整个电压曲线看一下结果"""
        file = 'Dynamic model data-20s/Data 0614/TJ-2021' + date + '.csv'
        df = pandas.read_csv(file)
        V = list(df['V'])
        T_out = list(df['Tout'])
        dV_static_star = list(df['dV_static_star'])
        v0 = V[num_step-1]
        T0 = T_out[num_step-1]  # 作为复原的起始值

        V = V[num_step:]
        T_out = T_out[num_step:]
        dV_static_star = dV_static_star[num_step:]
        return v0,T0,dV_static_star,V,T_out



