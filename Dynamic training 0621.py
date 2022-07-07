import time

import Dynamic_data_feed_0614 as datafeed
import Dynamic_model_0614 as models
import argparse
import pandas
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras


batch_size = 50

parser = argparse.ArgumentParser(description='processing integers.')
parser.add_argument('--mode', default = 'train', help = 'train or test')
parser.add_argument('--num_epochs', default= 61)
parser.add_argument('--batch_size', default = 50)
parser.add_argument('--learning_rate', default=1E-2)
parser.add_argument('--num_step', default=60)

args = parser.parse_args()

save = 1


t00 = time.time()
train = 10
if train == 1:
    dynamic_data = datafeed.dynamic_data()
    model = models.V_4LSTM()
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    for epoch_id in range(args.num_epochs):
        t0 = time.time()
        loss_epoch = 0
        # num_batches = dynamic_data.total_length//batch_size
        num_batches = 1920 // batch_size
        for batch_id in range(num_batches):
            X,Y = dynamic_data.get_easy_batch(num_step= args.num_step)
            with tf.GradientTape() as tape:
                Pred = model(X)
                loss = tf.reduce_sum(keras.losses.mean_squared_error(Y, Pred))
                # loss =tf.reduce_sum( keras.losses.mean_squared_logarithmic_error(Y,Pred))
            grads = tape.gradient(loss,model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))
            loss_epoch+= loss.numpy()
            # print(loss_epoch)
        X, v, t = dynamic_data.get_redemo_data()
        v_pred = []
        t_pred = []
        for idx in range(len(X)):
            cur_input = np.array(X[idx])
            cur_input = np.expand_dims(cur_input,axis=0)
            pred = model(cur_input)
            v_pred.append(pred[0,0])
            t_pred.append(pred[0,1])
        print('Epoch %d||cur loss %f||epoch time %f' % (epoch_id,loss_epoch,time.time() -t0))
        if epoch_id%10 == 0:
            plt.figure(figsize=(15, 8))  # 每10个epoch采样输出一个
            plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
            ax1 = plt.gca()
            ax1.plot(v)
            ax1.plot(v_pred,'red')
            ax1.legend(['original V', 'recover V'])
            ax1.set_ylim([-0.6,0.6])
            ax2 = ax1.twinx()
            ax2.scatter(range(len(t)),t,alpha = 0.3)
            ax2.plot(t_pred,'black')
            ax2.legend(['original T','revoer T'], loc=8)
            ax2.set_ylim([-0.6, 0.6])
            plt.title('Model training epoch %d, loss %f'%(epoch_id,loss_epoch))
        if save == 1:
            stor_file = 'Neural Networks/Dynamic model/Version 0621/trial 4 lstm 1.9.ckpt'
            model.save(stor_file)
    plt.show()



read = 1

if read == 1:
    model = keras.models.load_model('Neural Networks/Dynamic model/Version 0621/trial 4 lstm 1.9.ckpt')
recover = 1

if recover ==1:
    dynamic_data = datafeed.dynamic_data()

    X, v, t = dynamic_data.get_redemo_data(date = '1130')
    v0,T0,dV_static_star,V_ori,T_out_ori = dynamic_data.print_ori_v_t(date = '1130')

    v_pred = [v0]
    t_pred = [T0]
    for idx in range(len(X)):
        cur_input = np.array(X[idx])
        cur_input = np.expand_dims(cur_input, axis=0)
        pred = model(cur_input)

        v_pred.append(v_pred[-1] + dV_static_star[idx] + pred[0, 0] )
        t_pred.append( t_pred[-1] + pred[0, 1] )

    plt.figure(figsize=(15, 8))  # 每10个epoch采样输出一个
    plt.subplots_adjust(left=0.063, bottom=0.062, right=0.95, top=0.925)
    ax1 = plt.gca()
    ax1.plot(V_ori)
    ax1.plot(v_pred,'red')
    ax1.legend(['original V', 'recover V'])
    ax1.set_ylabel('voltage')
    ax2 = ax1.twinx()
    ax2.scatter(range(len(t)),T_out_ori,alpha = 0.3)
    ax2.plot(t_pred,'black')
    ax2.legend(['original dT','revoer dT'], loc=8)
    ax2.set_ylabel('temperature')
    plt.title('recovered voltage and temp')
    plt.show()
