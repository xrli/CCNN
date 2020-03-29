import numpy as np
import keras.backend as K
from keras.layers import *
from keras.models import Model, load_model, Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
from keras.optimizers import Adam
import keras_metrics
from sklearn.utils import class_weight

n_epochs = 100
batch_size = 64
lr = 1e-3



def HCCNN():
    sumprof_model = Sequential()
    sumprof_model.add(Conv1D(input_shape=(None, 1),
                             filters=16,
                             kernel_size=8,
                             activation='relu',
                             padding='valid',
                             kernel_initializer='glorot_uniform',
                             data_format='channels_last',
                             name='sumprof_conv1'))
    sumprof_model.add(Conv1D(filters=32,
                             kernel_size=8,
                             activation='relu',
                             padding='valid',
                             kernel_initializer='glorot_uniform',
                             data_format='channels_last',
                             name='sumprof_conv2'))
    sumprof_model.add(Conv1D(filters=64,
                             kernel_size=8,
                             activation='relu',
                             padding='valid',
                             kernel_initializer='glorot_uniform',
                             data_format='channels_last',
                             name='sumprof_conv3'))
    sumprof_model.add(GlobalMaxPooling1D(name='sumprof_pool'))
    sumprof_input = Input(shape=(None, 1))
    sumprof_output = sumprof_model(sumprof_input)


    subbands_model = Sequential()
    subbands_model.add(Conv2D(input_shape=(None, None, 1),
                              filters=16,
                              kernel_size=(4, 4),
                              activation='relu',
                              padding='valid',
                              kernel_initializer='glorot_uniform',
                              data_format='channels_last',
                              name='subbands_conv1'))
    subbands_model.add(Conv2D(filters=32,
                              kernel_size=(4, 4),
                              activation='relu',
                              padding='valid',
                              kernel_initializer='glorot_uniform',
                              data_format='channels_last',
                              name='subbands_conv2'))
    subbands_model.add(Conv2D(filters=64,
                              kernel_size=(4, 4),
                              activation='relu',
                              padding='valid',
                              kernel_initializer='glorot_uniform',
                              data_format='channels_last',
                              name='subbands_conv3'))
    subbands_model.add(GlobalMaxPooling2D(name='subbands_pool'))
    subbands_input = Input(shape=(None, None, 1))
    subbands_output = subbands_model(subbands_input)


    time_vs_phase_model = Sequential()
    time_vs_phase_model.add(Conv2D(input_shape=(None, None, 1),
                                   filters=16,
                                   kernel_size=(4, 4),
                                   activation='relu',
                                   padding='valid',
                                   kernel_initializer='glorot_uniform',
                                   data_format='channels_last',
                                   name='time_vs_phase_conv1'))
    time_vs_phase_model.add(Conv2D(filters=32,
                                   kernel_size=(4, 4),
                                   activation='relu',
                                   padding='valid',
                                   kernel_initializer='glorot_uniform',
                                   data_format='channels_last',
                                   name='time_vs_phase_conv2'))
    time_vs_phase_model.add(Conv2D(filters=64,
                                   kernel_size=(4, 4),
                                   activation='relu',
                                   padding='valid',
                                   kernel_initializer='glorot_uniform',
                                   data_format='channels_last',
                                   name='time_vs_phase_conv3'))
    time_vs_phase_model.add(GlobalMaxPooling2D(name='time_vs_phase_pool'))
    time_vs_phase_input = Input(shape=(None, None, 1))
    time_vs_phase_output = time_vs_phase_model(time_vs_phase_input)


    DM_model = Sequential()
    DM_model.add(Conv1D(input_shape=(None, 1),
                        filters=16,
                        kernel_size=16,
                        activation='relu',
                        padding='valid',
                        kernel_initializer='glorot_uniform',
                        data_format='channels_last',
                        name='DM_conv1'))
    DM_model.add(Conv1D(filters=32,
                        kernel_size=16,
                        strides=4,
                        activation='relu',
                        padding='valid',
                        kernel_initializer='glorot_uniform',
                        data_format='channels_last',
                        name='DM_conv2'))
    DM_model.add(Conv1D(filters=64,
                        kernel_size=8,
                        activation='relu',
                        padding='valid',
                        kernel_initializer='glorot_uniform',
                        data_format='channels_last',
                        name='DM_conv3'))
    DM_model.add(GlobalMaxPooling1D(name='DM_pool'))
    DM_input = Input(shape=(None, 1))
    DM_output = DM_model(DM_input)


    merged = concatenate([sumprof_output, subbands_output, time_vs_phase_output, DM_output], name='concat')

    
    dense1 = Dense(32, 
                   activation='relu',
                   kernel_initializer='glorot_uniform', 
                   name='dense1')(merged)

    output = Dense(1,
                   activation='sigmoid',
                   kernel_initializer='glorot_uniform',
                   name='output')(dense1) 
    model = Model(inputs=[sumprof_input, subbands_input, time_vs_phase_input, DM_input], outputs=output)
    adam = Adam(lr=lr)
    model.compile(loss='binary_crossentropy', 
                  optimizer=adam, 
                  metrics=['accuracy', 
                           keras_metrics.precision(), 
                           keras_metrics.recall()])
    return model


def VCCNN():
    sumprof_model = Sequential()
    sumprof_model.add(Conv1D(input_shape=(None, 1),
                             filters=16,
                             kernel_size=8,
                             activation='relu',
                             padding='valid',
                             kernel_initializer='glorot_uniform',
                             data_format='channels_last',
                             name='sumprof_conv1'))
    sumprof_model.add(Conv1D(filters=32,
                             kernel_size=8,
                             activation='relu',
                             padding='valid',
                             kernel_initializer='glorot_uniform',
                             data_format='channels_last',
                             name='sumprof_conv2'))
    sumprof_model.add(Conv1D(filters=64,
                             kernel_size=8,
                             activation='relu',
                             padding='valid',
                             kernel_initializer='glorot_uniform',
                             data_format='channels_last',
                             name='sumprof_conv3'))
    sumprof_model.add(GlobalMaxPooling1D(name='sumprof_pool'))
    sumprof_input = Input(shape=(None, 1))
    sumprof_output = sumprof_model(sumprof_input)


    subbands_model = Sequential()
    subbands_model.add(Conv2D(input_shape=(None, None, 1),
                              filters=16,
                              kernel_size=(4, 4),
                              strides=(2,2),
                              activation='relu',
                              padding='valid',
                              kernel_initializer='glorot_uniform',
                              data_format='channels_last',
                              name='subbands_conv1'))
    subbands_model.add(Conv2D(filters=32,
                              kernel_size=(4, 4),
                              activation='relu',
                              padding='valid',
                              strides=(2,2),
                              kernel_initializer='glorot_uniform',
                              data_format='channels_last',
                              name='subbands_conv2'))
    subbands_model.add(Conv2D(filters=64,
                              kernel_size=(4, 4),
                              activation='relu',
                              padding='valid',
                              strides=(2,2),
                              kernel_initializer='glorot_uniform',
                              data_format='channels_last',
                              name='subbands_conv3'))
    subbands_model.add(GlobalMaxPooling2D(name='subbands_pool'))
    subbands_input = Input(shape=(None, None, 1))
    subbands_output = subbands_model(subbands_input)


    time_vs_phase_model = Sequential()
    time_vs_phase_model.add(Conv2D(input_shape=(None, None, 1),
                                   filters=16,
                                   kernel_size=(4, 4),
                                   activation='relu',
                                   strides=(2,2),
                                   padding='valid',
                                   kernel_initializer='glorot_uniform',
                                   data_format='channels_last',
                                   name='time_vs_phase_conv1'))
    time_vs_phase_model.add(Conv2D(filters=32,
                                   kernel_size=(4, 4),
                                   strides=(2,2),
                                   activation='relu',
                                   padding='valid',
                                   kernel_initializer='glorot_uniform',
                                   data_format='channels_last',
                                   name='time_vs_phase_conv2'))
    time_vs_phase_model.add(Conv2D(filters=64,
                                   kernel_size=(4, 4),
                                   activation='relu',
                                   strides=(2,2),
                                   padding='valid',
                                   kernel_initializer='glorot_uniform',
                                   data_format='channels_last',
                                   name='time_vs_phase_conv3'))
    time_vs_phase_model.add(GlobalMaxPooling2D(name='time_vs_phase_pool'))
    time_vs_phase_input = Input(shape=(None, None, 1))
    time_vs_phase_output = time_vs_phase_model(time_vs_phase_input)


    DM_model = Sequential()
    DM_model.add(Conv1D(input_shape=(None, 1),
                        filters=16,
                        kernel_size=16,
                        activation='relu',
                        padding='valid',
                        kernel_initializer='glorot_uniform',
                        data_format='channels_last',
                        name='DM_conv1'))
    DM_model.add(Conv1D(filters=32,
                        kernel_size=16,
                        strides=4,
                        activation='relu',
                        padding='valid',
                        kernel_initializer='glorot_uniform',
                        data_format='channels_last',
                        name='DM_conv2'))
    DM_model.add(Conv1D(filters=64,
                        kernel_size=8,
                        activation='relu',
                        padding='valid',
                        kernel_initializer='glorot_uniform',
                        data_format='channels_last',
                        name='DM_conv3'))
    DM_model.add(GlobalMaxPooling1D(name='DM_pool'))
    DM_input = Input(shape=(None, 1))
    DM_output = DM_model(DM_input)


    merged = concatenate([Reshape((1, 64))(sumprof_output), 
                          Reshape((1, 64))(subbands_output), 
                          Reshape((1, 64))(time_vs_phase_output), 
                          Reshape((1, 64))(DM_output)
                          ], name='concat', axis=-2)
    merged = Reshape((4,64,1))(merged)

    conv1 = Conv2D(filters=16,
                   kernel_size=(3, 16),
                   padding='same',
                   strides=(2, 4),
                   kernel_initializer='glorot_uniform',
                   data_format='channels_last',
                   name='conv1')(merged)
    conv1 = Activation('relu')(conv1)

    conv1 = Conv2D(filters=2,
                   kernel_size=(3, 16),
                   padding='same',
                   strides=(2, 4),
                   kernel_initializer='glorot_uniform',
                   data_format='channels_last',
                   name='conv2')(conv1)
    conv1 = GlobalAveragePooling2D()(conv1)
    output = Activation('softmax')(conv1)

    model = Model(inputs=[sumprof_input, subbands_input, time_vs_phase_input, DM_input], outputs=output)
    adam = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=adam, 
                  metrics=['accuracy', 
                           keras_metrics.precision(), 
                           keras_metrics.recall()])

    return model



def training(model, data, y, verbose=0, logs='', model_mode='HCCNN'):
  if len(logs) == 0:
    callBacks=[]
  
  else:
    tbCallBack = TensorBoard(log_dir='logs',  
                 histogram_freq=0,
#                  batch_size=32,
                 write_graph=True, 
                 write_grads=True,
                 write_images=True,
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
    callBacks=[tbCallBack]


  class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y),
                                                  y)  
  if model_mode == 'VCCNN':
    y = keras.utils.to_categorical(y)
  model.fit([np.stack(data.sumprof)[:, :, np.newaxis],
             np.stack(data.subbands.values)[:, :, :, np.newaxis],
             np.stack(data.time_vs_phase)[:, :, :, np.newaxis],
             np.stack(data.DM)[:, :, np.newaxis]],
           y,
           class_weight=dict(enumerate(class_weights)),
           batch_size=batch_sz,
           epochs=n_epochs,
           shuffle=True,
           verbose=verbose,
           callbacks=[tbCallBack])

  return model


def test(model, data):
  y_prod = model.predict([np.stack(data.sumprof)[:, :, np.newaxis],
           np.stack(data.subbands.values)[:, :, :, np.newaxis],
           np.stack(data.time_vs_phase)[:, :, :, np.newaxis],
           np.stack(data.DM)[:, :, np.newaxis]])

  return y_prod