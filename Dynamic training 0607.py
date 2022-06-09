import argparse
import time
import pandas

import Dynamic_model_0527
import Dynamic_data_feed_0527

import numpy as np
import tensorflow as tf
import keras





# print(data.get_batch())
save = 1
parser = argparse.ArgumentParser(description='processing integers.')
parser.add_argument('--mode', default = 'train', help = 'train or test')
parser.add_argument('--num_epochs', default= 50)
parser.add_argument('--batch_size', default = 50)
parser.add_argument('--learning_rate', default=1E-2)
parser.add_argument('--num_step', default=60)
args = parser.parse_args()

print(args)

batch_size = args.batch_size
num_epochs = args.num_epochs

model_2lstm = Dynamic_model_0527.V_2LSTM()
optimizer = keras.optimizers.Adam(learning_rate = args.learning_rate)

read = 0
if read == 1:
    model_2lstm = keras.models.load_model('Neural Networks/Dynamic model/Verison1/0607-version2lstm 1.3.ckpt')


t00 = time.time()
train = 1
if train == 1:
    data = Dynamic_data_feed_0527.V_dynamic_data(batch_size = batch_size, source_folder= 'Dynamic model data-20s/Data 0525-dupli')
    for epoch_id in range(num_epochs):
        t0 = time.time()
        loss_epoch = 0
        num_batches = data.total_length // batch_size
        for batch_id in range(num_batches):
            x, y = data.get_batch(num_step= args.num_step)
            with tf.GradientTape() as tape:
                v_dynamic_pred = model_2lstm(x)
                loss = tf.reduce_sum(tf.square(v_dynamic_pred - y))
            grads = tape.gradient(loss, model_2lstm.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads,model_2lstm.variables))
            loss_epoch += loss
            print('batch %d/%d, loss = %f' % (batch_id, num_batches,loss))
        print('-------------------------------------------------------------')
        print('epoch %d, loss = %f, time = %f' % (epoch_id, loss_epoch,time.time()-t0))

        storage_file = 'Neural Networks/Dynamic model/Verison1/0607-version2lstm 1.5.ckpt'
        if save == 1:
            model_2lstm.save(storage_file)
print('total time consumption %f' % (time.time()-t00))

