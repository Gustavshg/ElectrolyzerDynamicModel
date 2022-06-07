import pandas
import time
import Dynamic_model_0527
import Dynamic_data_feed_0527

import numpy as np
import tensorflow as tf
import keras

batch_size = 50
num_epochs = 50

data = Dynamic_data_feed_0527.V_dynamic_data(batch_size)


# print(data.get_batch())
save = 1

model_lstm = Dynamic_model_0527.V_3LSTM()
model_lstm.compile(loss='mean_squared_error', optimizer='Adam')
optimizer = keras.optimizers.Adam(learning_rate=1E-3)

train = 1
if train == 1:
    for epoch_id in range(num_epochs):
        t0 = time.time()
        loss_epoch = 0
        num_batches = data.total_length // batch_size
        for batch_id in range(num_batches):
            x, y = data.get_batch()
            with tf.GradientTape() as tape:
                v_dynamic_pred = model_lstm(x)
                loss = tf.reduce_sum(tf.square(v_dynamic_pred - y))
            grads = tape.gradient(loss, model_lstm.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model_lstm.variables))
            loss_epoch += loss
            print('batch %d/%d, loss = %f' % (batch_id, num_batches, loss))
        print('-------------------------------------------------------------')
        print('epoch %d, loss = %f, time = %f' % (epoch_id, loss_epoch, time.time() - t0))

        storage_file = 'Neural Networks/Dynamic model/Verison1/0607-version3lstm 1.5.ckpt'
        if save == 1:
            model_lstm.save(storage_file)


