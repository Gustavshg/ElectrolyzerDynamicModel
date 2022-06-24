import time

import Dynamic_data_feed_0623 as datafeed
import Dynamic_model_Attention_0622 as models
import argparse
import pandas
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras


parser = argparse.ArgumentParser(description='processing integers.')
parser.add_argument('--mode', default = 'train', help = 'train or test')
parser.add_argument('--num_epochs', default= 151)
parser.add_argument('--batch_size', default = 1000)
parser.add_argument('--input_dims', default=7)
parser.add_argument('--num_step', default=32)
parser.add_argument('--learning_rate', default=1E-3)
parser.add_argument('--save', default=1)
parser.add_argument('--train', default=10)  # 当前是否训练
parser.add_argument('--recover', default=10)  # 是否需要复现数据
parser.add_argument('--model_redemo', default=1)  # 是否需要在虚拟数据上复现模型
args = parser.parse_args()




t00 = time.time()

if args.train == 1:
    dynamic_data = datafeed.dynamic_data(batch_size=args.batch_size)
    model = models.Only_LSTM(time_steps=args.num_step, input_dims= args.input_dims)
    model = model.get_model()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    # optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    # model.summary()
    from keras.utils import plot_model
    plot_model(model,to_file='Cache/model4.png',dpi = 300)

    X,Y = dynamic_data.get_model_fit_data(num_step= args.num_step)
    hist = keras.callbacks.TensorBoard(log_dir = 'Cache/hist',histogram_freq = 0, write_graph = True, write_images = True)
    model.fit(X,Y,batch_size=args.batch_size, epochs = args.num_epochs,validation_split= 0.2, verbose = 2, callbacks= [hist])

    if args.save == 1:
        stor_file = 'Neural Networks/Dynamic model/Version 0623/trial only lstm 1.1.ckpt'
        model.save(stor_file)
    print('Model saved at ' + stor_file)



if args.recover ==1:
    # 复现结果
    t0 = time.time()
    model = keras.models.load_model('Neural Networks/Dynamic model/Version 0623/trial only lstm 1.1.ckpt')
    model.summary()
    # from keras.utils import plot_model
    # plot_model(model,to_file = 'Cache/model1.png',show_shapes = True,show_layer_names = True,dpi = 300,show_layer_activations = True)
    dynamic_data = datafeed.dynamic_data()

    X, V_ori, T_ori = dynamic_data.get_redemo_data(num_step= args.num_step,date = '1129')

    v_pred = []
    t_pred = []
    for idx in range(len(X)):
        cur_input = np.array(X[idx])
        cur_input = np.expand_dims(cur_input, axis=0)
        pred = model(cur_input)
        v_pred.append( pred[0, 0] )
        t_pred.append( pred[0, 1] )

    plt.figure(figsize=(15, 8))  # 每10个epoch采样输出一个
    plt.subplots_adjust(left=0.063, bottom=0.062, right=0.95, top=0.925)
    ax1 = plt.gca()
    ax1.plot(V_ori)
    ax1.plot(v_pred,'red')
    ax1.legend(['original V', 'recover V'])
    ax1.set_ylabel('voltage')
    ax2 = ax1.twinx()
    ax2.scatter(range(len(T_ori)),T_ori,alpha = 0.3)
    ax2.plot(t_pred,'black')
    ax2.legend(['original dT','revoer dT'], loc=8)
    ax2.set_ylabel('temperature')
    plt.title('recovered voltage and temp')
    print(time.time()-t0)
    plt.show()



if args.model_redemo == 1:
    """这部分测试模型能否在虚构的数据上正确运行"""
    import Smoothen as sm
    dynamic_data = datafeed.dynamic_data()
    X, V_ori, T_ori = dynamic_data.get_redemo_data(num_step=args.num_step, date='1129')

    v_pred = [0]
    t_pred = [0]
    model = keras.models.load_model( 'Neural Networks/Dynamic model/Version 0623/trial attention lstm double simplified 32 step no compile and fit 1.7.ckpt')
    model.summary()
    for idx in range(len(X)):
        cur_input = np.array(X[idx])
        cur_input = np.expand_dims(cur_input, axis=0)
        pred = model(cur_input)
        v_pred.append( pred[0, 0] )
        t_pred.append( pred[0, 1] )
    dV,dTout = dynamic_data.print_ori_v_t(num_step=args.num_step, date='1129')

    dV_pred = sm.diff(v_pred)
    dT_pred = sm.diff(t_pred)
    plt.figure(figsize=(15, 8))  # 每10个epoch采样输出一个
    plt.subplots_adjust(left=0.063, bottom=0.062, right=0.95, top=0.925)
    ax1 = plt.gca()
    ax1.plot(dV_pred,'red')
    ax1.scatter(range(len(dV)),dV,alpha = 0.25)
    ax1.legend(['revocered-dV','dV'])
    ax1.set_ylabel('voltage')
    ax2 = ax1.twinx()
    ax2.plot(dT_pred,'red')
    ax2.scatter(range(len(dTout)),dTout,alpha = 0.25)
    ax2.legend(['revocered-dT','dT'], loc=8)
    ax2.set_ylabel('temperature')
    plt.title('recovered voltage and temp')
    plt.show()



































