


# ================================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import scipy.io
from scipy.stats import truncnorm
from tensorflow.keras.layers import Input, concatenate, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, BatchNormalization, Dropout, Activation, UpSampling2D, Conv3DTranspose, Flatten, Dense, Reshape,LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU
from numpy import linalg as LA
# =============================================================================

class DQN(object):
    
    def save_model(self,save_file_path , save_file_name):
        self.model.save(os.path.join(save_file_path, save_file_name))

    def load_model(self, load_file_path, load_file_name):
        self.model=keras.models.load_model(os.path.join(load_file_path, load_file_name))

    def load_weights(self, load_file_path, load_file_name):
        self.model.load_weights(os.path.join(load_file_path, load_file_name))

    def print_model_layers(self):
        for i, layer in enumerate(self.model.layers):
            print("Layer", i, "\t", layer.name, "\t\t", layer.input_shape, "\t", layer.output_shape)

    def find_first_bit_set(self, num):
        # Check there is at least one bit set.
        if num == 0:
            return None
        elif num is None:
            return float('inf')
        # Right-shift until we have the first set bit in the LSB position.
        i = 0
        while (num % 2) == 0:
            i += 1
            num = num >> 1
        return i

    def l1_norm_accuracy(self,y_true,y_pred):
        a = abs(y_true-y_pred)
        b = K.mean(a)
        return b

    def scatter_removel(self,y_true, y_pred):
        a = K.mean(abs(y_true-y_pred*tf.log(tf.maximum(y_true,1e-6))))
        b = 10*self.l1_norm_accuracy(y_true, y_pred)
        return a+b

    def psnr_accuracy(self,y_true,y_pred):
        return 10.0 * tf.log(tf.reduce_max(y_true,axis=None) / tf.reduce_mean(tf.squared_difference(y_pred, y_true)))/ tf.log(10.0)
    def conv2d_bn(self,x,
                  filters,
                  num_row,
                  num_col,
                  padding='same',
                  strides=(1, 1, 1),
                  name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        x = Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding, 
            use_bias=False,
            name=conv_name)(x)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation('relu', name=name)(x)
        #x = Dropout(0.1)(x)
        #x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        return x

    def build_and_compile_DQN(self, img_x = None, channels_in = 5, channels_out=1, dropout_rate=0, learningRate = None):        # layer = {}

        layer = {}
        current_filter_num = 64
        layer[0] = Input((img_x, channels_in))
        drop_rate_layer = dropout_rate * (np.sqrt(np.sqrt((current_filter_num / 2048))))
        layer[1] = Dropout(rate=drop_rate_layer)(BatchNormalization()(LeakyReLU(alpha=0.1)(Conv1D(current_filter_num, 5, padding='valid')(layer[0]))))
        current_filter_num = current_filter_num*2
        layer[2] = MaxPooling1D(pool_size=2)(layer[1])
        layer[3] = Dropout(rate=drop_rate_layer)(BatchNormalization()(LeakyReLU(alpha=0.1)(Conv1D(current_filter_num, 5, padding='valid')(layer[2]))))
        current_filter_num = current_filter_num * 2
        layer[4] = MaxPooling1D(pool_size=2)(layer[3])
        layer[5] = Dropout(rate=drop_rate_layer)(BatchNormalization()(LeakyReLU(alpha=0.1)(Conv1D(current_filter_num, 3, padding='valid')(layer[4]))))
        current_filter_num = current_filter_num * 2
        layer[6] = MaxPooling1D(pool_size=2)(layer[5])
        layer[7] = Dropout(rate=drop_rate_layer)(BatchNormalization()(LeakyReLU(alpha=0.1)(Conv1D(current_filter_num, 3, padding='valid')(layer[6]))))
        layer[8] = MaxPooling1D(pool_size=2)(layer[7])
        layer[9] = Flatten()(layer[8])
        layer[10] = LeakyReLU(alpha=0.1)(Dense(512)(layer[9]))
        layer[11] = LeakyReLU(alpha=0.1)(Dense(64)(layer[10]))
        layer[12] = LeakyReLU(alpha=0.1)(Dense(3)(layer[11]))
        self.model = Model(inputs=[layer[0]], outputs=layer[12])
        self.model.compile(optimizer=Adam(lr=learningRate), loss='mae')
