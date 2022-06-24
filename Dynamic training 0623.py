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
parser.add_argument('--num_epochs', default= 51)
parser.add_argument('--batch_size', default = 1000)
parser.add_argument('--input_dims', default=8)
parser.add_argument('--num_step', default=32)
parser.add_argument('--learning_rate', default=1E-2)
parser.add_argument('--save', default=1)
parser.add_argument('--train', default=1)
parser.add_argument('--recover', default=10)
args = parser.parse_args()




t00 = time.time()

if args.train == 1:
    dynamic_data = datafeed.dynamic_data(batch_size=args.batch_size)
    model = models.Attention_LSTM_no_dV_static(time_steps=args.num_step, input_dims= args.input_dims)
    model = model.get_model()

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    # model.summary()
    # from keras.utils import plot_model
    # plot_model(model,to_file='Cache/model3.png',dpi = 300)
    for epoch_id in range(args.num_epochs):
        t0 = time.time()
        loss_epoch = 0
        num_batches = dynamic_data.total_length//args.batch_size+1
        for batch_id in range(num_batches):
            X,Y = dynamic_data.get_batch(num_step= args.num_step)

            with tf.GradientTape() as tape:
                Pred = model(X)
                loss =tf.reduce_sum( keras.losses.mean_squared_error(Y,Pred))
            grads = tape.gradient(loss,model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))
            loss_epoch+= loss.numpy()
            # print(loss_epoch)
        print('Epoch %d||cur loss %f||epoch time %f' % (epoch_id, loss_epoch, time.time() - t0))
        # X, v, t = dynamic_data.get_redemo_data()
        # v_pred = []
        # t_pred = []
        # for idx in range(len(X)):
        #     cur_input = np.array(X[idx])
        #     cur_input = np.expand_dims(cur_input,axis=0)
        #     pred = model(cur_input)
        #     v_pred.append(pred[0,0])
        #     t_pred.append(pred[0,1])

        # if epoch_id%10 == 0:
        #     plt.figure(figsize=(15, 8))  # 每10个epoch采样输出一个
        #     plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
        #     ax1 = plt.gca()
        #     ax1.plot(v)
        #     ax1.plot(v_pred,'red')
        #     ax1.legend(['original V', 'recover V'])
        #     ax1.set_ylim([-0.6,0.6])
        #     ax2 = ax1.twinx()
        #     ax2.scatter(range(len(t)),t,alpha = 0.3)
        #     ax2.plot(t_pred,'black')
        #     ax2.legend(['original T','revoer T'], loc=8)
        #     ax2.set_ylim([-0.6, 0.6])
        #     plt.title('Model training epoch %d, loss %f'%(epoch_id,loss_epoch))
        if args.save == 1 and epoch_id%10==0:
            stor_file = 'Neural Networks/Dynamic model/Version 0623/trial attention lstm double simplified 32 step no dV_static_star 1.1.ckpt'
            model.save(stor_file)
    print('Model saved at' + stor_file)
    # plt.show()


if args.recover ==1:
    # 复现结果
    t0 = time.time()
    model = keras.models.load_model('Neural Networks/Dynamic model/Version 0623/trial attention lstm double simplified 32 step 1.1.ckpt')
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
