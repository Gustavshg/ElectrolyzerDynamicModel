import Dynamic_data_feed_0630 as datafeed
import Dynamic_model_no_residual_0630 as modelsets
import argparse
import time
import pandas
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

parser = argparse.ArgumentParser(description='processing integers.')
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument('--num_epochs', default=300)
parser.add_argument('--batch_size', default=1000)
parser.add_argument('--input_dims', default=7)
parser.add_argument('--num_step', default=64)
parser.add_argument('--learning_rate', default=1E-5)
parser.add_argument('--save', default=1)  # 是否保存当前模型
parser.add_argument('--train', default=1)  # 当前是否训练
parser.add_argument('--recover', default=10)  # 是否需要复现数据
parser.add_argument('--model_redemo', default=10)  # 是否需要在虚拟数据上复现模型
parser.add_argument('--artificial_data', default=1)  # 是否使用虚拟数据进行验证
args = parser.parse_args()

t00 = time.time()

if args.train == 1:
    dynamic_data = datafeed.dynamic_data(batch_size=args.batch_size)
    model = modelsets.Attention_LSTM_spaced(time_steps=args.num_step, input_dims=args.input_dims,
                                                 output_dims=2)
    model = model.get_model()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    from keras.utils import plot_model
    plot_model(model, to_file='Cache/model6 0630.png', dpi=300)
    X,Y = dynamic_data.get_model_fit_data(num_step= args.num_step)
    hist = keras.callbacks.TensorBoard(log_dir = 'Cache/hist0630',histogram_freq = 0, write_graph = True, write_images = True)
    model.fit(X,Y,batch_size=args.batch_size, epochs = args.num_epochs,validation_split= 0.25, verbose = 2, callbacks= [hist])
    if args.save == 1:
        stor_file = 'Neural Networks/Dynamic model/Version 0630/Attention lstm spaced 64 1.5.ckpt'

        model.save(stor_file)
    print('Model saved at ' + stor_file)

if args.artificial_data == 1:
    import matplotlib.pyplot as plt
    model = keras.models.load_model(
        'Neural Networks/Dynamic model/Version 0630/Attention lstm spaced 64 1.5.ckpt')
    # model.summary()
    num_step = 64

    data = datafeed.dynamic_data()
    X,Y = data.get_batch(num_step = num_step)
    print(X.shape)
    print(X[0,-1,:],Y[0])
    # # 这里可以自己创建一个开机过程的基础数据
    # data = datafeed.Artificial_data(num_step=num_step)
    # V_star, T_out_star, Current_density, T_in, Lye_flow, Amb_temp, dJ = data.start_up(current_density_setpoint=3500,
    #                                                                                   v0=0.00027, t0=22.034, lye_time=300,
    #                                                                                   lye_flow=1.3, t_lye_0=22.181,
    #                                                                                   t_lye=61.5, wait_time=100,
    #                                                                                   close_time=2500, total_time=7000)
    # # 开始步进计算模型
    # for idx in range(len(Current_density) - num_step):
    #     V_next,T_next = datafeed.step(model,idx,num_step,V_star, T_out_star, Current_density, T_in, Lye_flow, Amb_temp, dJ )
    #     V_star.append(V_next * 2.5)
    #     T_out_star.append(T_next * 10)
    #     print(idx,V_next,T_next)
    # plt.figure()
    # plt.plot(V_star)
    # plt.title('V predicted')
    # plt.figure()
    # plt.plot(T_out_star)
    # plt.title('T predicted')
    # plt.show()

    # 这里可以读取一天的数据作为输入数据
    df = pandas.read_csv('Dynamic model data-20s/Data 0614/TJ-20210924.csv')
    V_star = list(df['V_star'])[:num_step]
    T_out_star = list(df['T_out_star'])[:num_step]
    Current_density = list(df['Current density'])
    T_in = list(df['Tlye'])
    Lye_flow = list(df['LyeFlow'])
    Amb_temp = list(df['AmbT'])
    dJ = list(df['dj'])
    # print(len(V_star))
    V_pred = []
    T_pred = []

    for idx in range(len(Current_density) - num_step):
        V_next,T_next = datafeed.step(model,idx,num_step,V_star, T_out_star, Current_density, T_in, Lye_flow, Amb_temp, dJ )
        V_star.append(V_next*2.2)
        T_out_star.append(T_next*85.)
        # print(idx,V_next,T_next)
    plt.figure(figsize=(15,8))
    plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
    plt.subplot(121)
    plt.plot(V_star)
    plt.plot(df['V'])
    plt.title('V predicted')
    plt.subplot(122)
    plt.plot(T_out_star)
    plt.plot(df['Tout'])
    plt.title('T predicted')
    plt.show()