import argparse

import pandas
import time
import Dynamic_model_0527
import Dynamic_data_feed_0527
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
plt.style.use('seaborn')
batch_size = 50
num_epochs = 50

parser = argparse.ArgumentParser(description='processing integers.')
parser.add_argument('--mode', default = 'train', help = 'train or test')
parser.add_argument('--num_epochs', default= 50)
parser.add_argument('--batch_size', default = 50)
parser.add_argument('--learning_rate', default=1E-2)
parser.add_argument('--num_step', default=60)
args = parser.parse_args()

# print(data.get_batch())
save = 1

# model_lstm = keras.models.load_model('Neural Networks/Dynamic model/Verison1/0607-version 1.2.ckpt')
# model_lstm.summary()
model_lstm = keras.models.load_model( 'Neural Networks/Dynamic model/Verison1/0607-version3lstm 1.9.ckpt')
model_lstm.summary()
df = pandas.read_csv('Dynamic model data-20s/Data 0525-dupli/TJ-20211129.csv')
T_out = df['Tout']
Current_density = df['Current density']
T_in = df['Tlye']
Lye_flow = df['LyeFlow']
V_static = df['V_static']
V_dynamic = df['V_dynamic']

print(df.columns)
# plt.figure()
# plt.plot(T_in)


V_pred = []
x = np.zeros([1,args.num_step,5])
pred =1
if pred == 1:
    for j in range(len(df)):
        print(j)
        if j < args.num_step+1:
            V_pred.append(-1.2)
        else:
            """这里是每次计算一个点就可以了"""
            cur_T_out = T_in[j - args.num_step - 1:j - 1]  # 出口温度因为是需要预测的，所以需要上一时刻出口温度，因为当前采样时刻的出口温度不知道
            cur_Current_density = Current_density[j - args.num_step:j]
            cur_T_in = T_in[j - args.num_step:j]
            cur_Lye_flow = Lye_flow[j - args.num_step:j]
            cur_V_static = V_static[j - args.num_step:j]
            # cur_V_dynamic = V_dynamic[j - num_step:j]  # 这个是拟合的指标
            cur_V_dynamic = V_dynamic[j]  # 这个是拟合的指标，只选取当前时刻作为结果，而不是整个seq
            cur_input = np.array([cur_T_out, cur_T_in, cur_Current_density, cur_Lye_flow, cur_V_static])
            cur_input = cur_input.T
            cur_input = np.expand_dims( cur_input, axis= 0)
            x = cur_input


            v_pred = model_lstm(x)
            v_pred = np.array(v_pred)
            for scaler in v_pred:
                V_pred.append(float(scaler))
            """感觉根据0607晚上的想法，这里其实每一个batch就是一个预测点，每个batch都选取最后的结果，我们这里为了省事儿，其实是吧训练函数拿来做预测函数用了，严谨的方式还是应该再写一个预测函数，这个需要后面做一下"""
            """我们之前在写极化曲线的时候其实没有用到batch，所以这次需要学习一下"""
            # if len(x)<=2:
            #     cur_T_out = T_in[j - args.num_step - 1:j - 1]  # 出口温度因为是需要预测的，所以需要上一时刻出口温度，因为当前采样时刻的出口温度不知道
            #     cur_Current_density = Current_density[j - args.num_step:j]
            #     cur_T_in = T_in[j - args.num_step:j]
            #     cur_Lye_flow = Lye_flow[j - args.num_step:j]
            #     cur_V_static = V_static[j - args.num_step:j]
            #     # cur_V_dynamic = V_dynamic[j - num_step:j]  # 这个是拟合的指标
            #     cur_V_dynamic = V_dynamic[j]  # 这个是拟合的指标，只选取当前时刻作为结果，而不是整个seq
            #     cur_input = np.array([cur_T_out, cur_T_in, cur_Current_density, cur_Lye_flow, cur_V_static])
            #     cur_input = cur_input.T
            #     cur_input = np.expand_dims( cur_input, axis= 0)
            #     x = np.concatenate([x,cur_input], axis= 0)
            #
            # else:
            #     x = x[1:,:,:]
            #     v_pred = model_lstm(x)
            #     v_pred = np.array(v_pred)
            #     for scaler in v_pred:
            #         V_pred.append(float(scaler))
            #     x = np.zeros([1, args.num_step, 5])
            """上面的是需要按照batch来的，这里我们不用batch，因为发现好像输入任意长度batch都是可以的，那就一个点一个点来吧"""



    plt.figure(figsize=(15, 8))
    ax1 = plt.gca()
    ax1.plot(V_pred)
    ax1.plot(V_dynamic)
    ax1.legend(['Dynamic model', 'original voltage'])
    ax2 = ax1.twinx()
    ax2.plot(Current_density,'r')
    ax2.legend(['current density'],loc = 8)
    plt.title('3 layer of gru')


plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)


plt.show()