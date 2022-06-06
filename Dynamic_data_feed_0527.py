import pandas
import numpy as np
import os


class V_dynamic_data():
    """这个里面是比较简单的模型数据的尝试，就是简单的五个数据内容"""
    def __init__(self):
        self.source_folder = 'Dynamic model data-20s/Data 0525'
        self.file_list = os.listdir(self.source_folder)
        self.file_list.sort()
        self.file_num = 5
        self.batch_file_list = np.random.randint(low=0, high=len(self.file_list), size=self.file_num)  # 随机抽取5个文件，每个文件里抽取10个sequence
        ALL_columns = ['时间', '电解电压', '电解电流', '产氢量', '产氢累计量', '碱液流量', '碱温',
                       '系统压力  ', '氧槽温', '氢槽温', '氧侧液位', '氢侧液位', '氧中氢', '氢中氧', '脱氧上温', '脱氧下温',
                       'B塔上温', 'B塔下温', 'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', '露点', '微氧量', '出罐压力',
                       '进罐温度', '进罐压力', 'V', 'I', 'Current density', 'Pressure', 'Tlye', 'TH2',
                       'TO2', 'Tout', 'LyeFlow', 'LyeFlow_Polar', 'dI', 'dj', 'dV', 'dTout',
                       'HTO', 'OTH', 'polar_nn', 'polar_lh', 'polar_wtt', 'polar_shg',
                       'dTout_WL', 'V_static', 'V_dynamic']

    def get_batch(self,batch_size = 50,num_step = 40):
        X = []
        Y = []
        for i in range(len(self.batch_file_list)):
            file = self.batch_file_list[i]
            file = self.file_list[file]
            print(file)
            file = os.path.join(self.source_folder, file)
            df = pandas.read_csv(file)
            T_out = df['Tout']
            Current_density = df['Current density']
            T_in = df['Tlye']
            Lye_flow = df['LyeFlow']
            V_static = df['V_static']
            V_dynamic = df['V_dynamic']
            random_locs = np.random.randint(low=num_step + 1, high=len(df),
                                            size=batch_size // self.file_num)  # 这里需要从51开始，因为不然这样就没办法抽取上一个时刻的出口温度
            for j in random_locs:
                cur_T_out = T_in[j - num_step - 1:j - 1]  # 出口温度因为是需要预测的，所以需要上一时刻出口温度，因为当前采样时刻的出口温度不知道
                cur_Current_density = Current_density[j - num_step:j]
                cur_T_in = T_in[j - num_step:j]
                cur_Lye_flow = Lye_flow[j - num_step:j]
                cur_V_static = V_static[j - num_step:j]
                cur_V_dynamic = V_dynamic[j - num_step:j]  # 这个是拟合的指标
                X.append([cur_T_out, cur_T_in, cur_Current_density, cur_Lye_flow, cur_V_static])
                Y.append(cur_V_dynamic)
        X = np.array(X).transpose([0, 2, 1])  # 最后会是[batch_size,num_step,features]的格式
        Y = np.array(Y).reshape(batch_size, num_step, 1)
        return X,Y


Vd_data = V_dynamic_data()
inputs,outputs = Vd_data.get_batch()
print(outputs.shape)


