import argparse

import pandas
import time
import Dynamic_model_0527
import Dynamic_data_feed_0527
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

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
model_lstm = keras.models.load_model('Neural Networks/Dynamic model/Verison1/0607-version2lstm 1.3.ckpt')
model_lstm.summary()
df = pandas.read_csv('Dynamic model data-20s/Data 0525-dupli/TJ-20211007.csv')
T_out = df['Tout']
Current_density = df['Current density']
T_in = df['Tlye']
Lye_flow = df['LyeFlow']
V_static = df['V_static']
V_dynamic = df['V_dynamic']

plt.figure()
plt.plot(T_in)


V_pred = []
x = np.zeros([1,args.num_step,5])
pred = 0
if pred == 1:
    for j in range(len(df)):
        if j < args.num_step+1:
            V_pred.append(0.)
        else:
            if len(x)<=50:
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
                x = np.concatenate([x,cur_input], axis= 0)

            else:
                x = x[1:,:,:]
                v_pred = model_lstm(x)
                v_pred = np.array(v_pred)
                for scaler in v_pred:
                    V_pred.append(float(scaler))
                x = np.zeros([1, args.num_step, 5])
    plt.figure()
    plt.plot(V_pred)
    plt.plot(V_dynamic)
    plt.legend(['Dynamic model', 'original voltage'])
    plt.title('2 layer of gru')


plt.show()