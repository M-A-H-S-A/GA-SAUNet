import numpy
import numpy as np
import configparser
import random
import collections
import tensorflow as tf
from tensorflow import keras
from deap import base, creator, tools, algorithms
from keras.models import Model
from keras import layers
from keras.layers import Input, concatenate, UpSampling2D, Dropout, AveragePooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda, Conv1D, Layer, MaxPooling2D, AveragePooling2D, BatchNormalization, add, Conv2DTranspose
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
from matplotlib import pyplot as plt
import ast
import sys
import gc
import pickle
from keras_drop_block import DropBlock2D
import time
import datetime
#function to obtain data for training/testing (validation)
start_time = time.time()
print ("Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
import os
import cv2
from skimage import data, color
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

strategy = tf.distribute.MirroredStrategy()
name_net = "SA-UNet"

data_location = ''

training_images_loc = data_location + 'DRIVE/train/images/'
training_label_loc = data_location + 'DRIVE/train/labels/'

validate_images_loc = data_location + 'DRIVE/validate/images/'
validate_label_loc = data_location + 'DRIVE/validate/labels/'
train_files = os.listdir(training_images_loc)
train_data = []
train_label = []
validate_files = os.listdir(validate_images_loc)
validate_data = []
validate_label = []
desired_size = 592
for i in train_files:
    im = imread(training_images_loc + i)
    label = imread(training_label_loc + i.split('_')[0] + '_manual1.png',pilmode="L")
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    train_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    train_label.append(temp)

for i in validate_files:
    im = imread(validate_images_loc + i)
    label = imread(validate_label_loc + i.split('_')[0] + '_manual1.png',pilmode="L")
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    validate_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    validate_label.append(temp)

train_data = np.array(train_data)
train_label = np.array(train_label)

validate_data = np.array(validate_data)
validate_label = np.array(validate_label)

x_train = train_data.astype('float32') / 255.
y_train = train_label.astype('float32') / 255.
x_train = np.reshape(x_train, (
len(x_train), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_train = np.reshape(y_train, (len(y_train), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

x_validate = validate_data.astype('float32') / 255.
y_validate = validate_label.astype('float32') / 255.
x_validate = np.reshape(x_validate, (
len(x_validate), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_validate = np.reshape(y_validate,
                        (len(y_validate), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

TensorBoard(log_dir='./autoencoder', histogram_freq=0,
            write_graph=True, write_images=True)

def spatial_attention(input_feature):
    kernel_size=7

    if K.image_data_format() == "channels_first":
        channel=input_feature.shape[1]
        cbam_feature=Permute((2, 3, 1))(input_feature)
    else:
        channel=input_feature.shape[-1]
        cbam_feature=input_feature

    avg_pool=Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool=Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat=Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2

    cbam_feature=Conv2D(1, (7, 7),
                        strides=1,
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer='he_normal',
                        use_bias=False)(concat)
    print('---->' + str(cbam_feature.shape))
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature=Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)


def attention_gate(inp_1, inp_2, n_intermediate_filters):
    """Attention gate. Compresses both inputs to n_intermediate_filters filters before processing.
       Implemented as proposed by Oktay et al. in their Attention U-net, see: https://arxiv.org/abs/1804.03999.
    """
    inp_1_conv=Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        data_format=None,
        kernel_initializer="he_normal",
    )(inp_1)
    inp_2_conv=Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        data_format=None,
        kernel_initializer="he_normal",
    )(inp_2)

    f=Activation("relu")(add([inp_1_conv, inp_2_conv]))
    g=Conv2D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding="same",
        data_format=None,
        kernel_initializer="he_normal",
    )(f)
    h=Activation("sigmoid")(g)
    return multiply([inp_1, h])


def attention_concat(conv_below, skip_connection):
    """Performs concatenation of upsampled conv_below with attention gated version of skip-connection
    """
    below_filters=conv_below.get_shape().as_list()[-1]
    attention_across=attention_gate(skip_connection, conv_below, below_filters)
    return concatenate([conv_below, attention_across])


def conv2d_block(
        inputs,
        use_batch_norm=True,
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
):
    c=Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding='same',
        data_format=None,
        use_bias=not use_batch_norm,
    )(inputs)
    if use_batch_norm:
        c=BatchNormalization()(c)

    c=Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding='same',
        data_format=None,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c=BatchNormalization()(c)
    return c

def custom_unet(
        input_size,
        activation="sigmoid",
        use_batch_norm=True,
        upsample_mode="deconv",  # 'deconv' or 'simple'
        use_dropout_on_upsampling=False,
        use_attention=True,
        filters=16,
        kernel=(3, 3),
        network_depth=3,
        pooling_type=2,
        optimiser='RMSprop',
        keep_prob=0.8,
        block_size=7,
        output_activation="sigmoid",
):  # 'sigmoid' or 'softmax'

    if upsample_mode == "deconv":
        upsample=upsample_conv
    else:
        upsample=upsample_simple

    # Build U-Net model
    inputs=Input(input_size)
    x=inputs

    if (pooling_type == 1):
        pooling_2d=MaxPooling2D
    else:
        pooling_2d=AveragePooling2D

    down_layers=[]
    for l in range(network_depth):
        x=conv2d_block(
            inputs=x,
            filters=filters,
            kernel_size=(kernel[0], kernel[1]),
            use_batch_norm=use_batch_norm,
            activation=activation,
            padding='same',
        )
        down_layers.append(x)
        x=DropBlock2D(block_size=block_size, keep_prob=keep_prob)(x)
        x=pooling_2d((2, 2), padding='same', data_format=None)(x)
        filters=filters * 2  # double the number of filters with each layer

    convm=Conv2D(filters=filters, kernel_size=(kernel[0], kernel[1]), activation=activation, padding='same',
                 data_format=None)(x)
    convm=DropBlock2D(block_size=block_size, keep_prob=keep_prob)(convm)
    convm=BatchNormalization()(convm)
    convm=Activation('relu')(convm)
    convm=spatial_attention(convm)
    convm=Conv2D(filters=filters, kernel_size=(kernel[0], kernel[1]), activation=activation, padding='same',
                 data_format=None)(convm)
    convm=DropBlock2D(block_size=block_size, keep_prob=keep_prob)(convm)
    convm=BatchNormalization()(convm)
    x=Activation('relu')(convm)
    x=conv2d_block(
        inputs=x,
        filters=filters,
        activation=activation,
    )
    for conv in reversed(down_layers):
        filters//=2  # decreasing number of filters with each layer
        x=upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        if use_attention:
            x=attention_concat(conv_below=x, skip_connection=conv)
        else:
            x=concatenate(axis=1)([x, conv])
        x=DropBlock2D(block_size=block_size, keep_prob=keep_prob)(x)
        x=conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            activation=activation,
            padding='same',
        )

    outputs=Conv2D(1, (1, 1), padding="same", activation=activation, data_format=None)(x)


    model=Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimiser, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model





input_size=(desired_size,desired_size,3)




# ========= Load settings from Config file
config=configparser.RawConfigParser()
# config = ConfigParser.RawConfigParser()
config.read('configuration.txt')
# patch to the datasets
#path_data=config.get('data paths', 'path_local')
# Experiment name
name_experiment=config.get('experiment name', 'name')
# training settings

model=custom_unet(
        input_size,
        activation="sigmoid",
        use_batch_norm=True,
        upsample_mode="deconv",  # 'deconv' or 'simple'
        use_dropout_on_upsampling=False,
        use_attention=True,
        filters=16,
        kernel=(3, 3),
        network_depth=3,
        pooling_type=2,
        optimiser='RMSprop',
        keep_prob=0.8,
        block_size=7)

print( "Check: final output of the network:")
print( model.output_shape)
plt(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)



checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased

history=model.fit(x_train, y_train, epochs=150, batch_size=8, verbose=2, validation_data=(x_validate, y_validate), shuffle=True)
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)

def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'./'+name_experiment+'/'+name_experiment +'_accuracy.png')
    plt.clf()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
