import numpy
import numpy as np
import configparser
import random
import collections
import tensorflow as tf
from deap import base, creator, tools, algorithms
from keras.models import Model
from keras.layers import Input, concatenate, UpSampling2D, Dropout, AveragePooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda, Conv1D, Layer, MaxPooling2D, AveragePooling2D, BatchNormalization, add, Conv2DTranspose

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
import ast
import gc
from keras_drop_block import DropBlock2D
import time
import datetime
start_time = time.time()
print ("Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
import os
import cv2
from skimage.io import imread



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

strategy = tf.distribute.MirroredStrategy()
name_net = "SA-UNet"

# ====================Prepare dataset========================


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



# ====================SA========================


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
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature=Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

# ====================Model========================

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)


def attention_gate(inp_1, inp_2, n_intermediate_filters):

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

# ====================SA-model========================
def custom_unet(
        input_size,
        activation="relu",
        use_batch_norm=True,
        upsample_mode="deconv",  # 'deconv' or 'simple'
        use_attention=True,
        filters=16,
        kernel=(2, 2),
        network_depth=4,
        pooling_type=1,
        optimiser='sgd',
        keep_prob=0.9,
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

# ====================Hyper-parameters representation========================

design={'activation': 3, 'optimiser': 3, 'pooling_type': 1, 'kernel': 2, 'network_depth': 2, 'number_filter': 2, 'keep_prob': 1, 'block_size': 1}
network_depth_dict={'00': 1, '01': 2, '10': 3, '11': 4}
number_filter_dict={'00': 16, '01': 32, '11': 64, '10': 128}
kernel_dict={'00': (3, 3), '01': (3, 3), '10': (5, 5), '11': (7, 7)}
optimiser_dict={'000': 'sgd', '001': 'adam', '100': 'adamax', '110': 'adagrad', '111': 'Nadam', '101': 'Ftrl',
                '011': 'Adadelta', '010': 'RMSprop'}

pooling_type_dict={'0': '1', '1': '2'}
activation_dict={'000': 'relu', '001': 'sigmoid', '010': 'softmax', '110': 'softplus', '101': 'softsign', '111': 'tanh',
                 '011': 'selu', '100': 'elu'}
keep_prob_dict={'0': 0.9, '1': 0.8}
block_size_dict={'0': 7, '1': 9}
pheno_list=[activation_dict, optimiser_dict, pooling_type_dict, kernel_dict, network_depth_dict, number_filter_dict, keep_prob_dict, block_size_dict]


def to_genes(individual):
    print(f'In to_genes')
    chromosome=dict()
    for key, value in design.items():
        chromosome[key]=''.join(str(i) for i in individual[0:value])
        individual=individual[value::]
    return chromosome


def to_phenos(chromosome, design, pheno_list):
    print(f'In to_phenos')
    count=0
    count_list=0
    keys=list(design)
    pheno_type=dict()
    for key, value in chromosome.items():
        if (len([int(x) for x in list(value)]) <= len(max(list(i.keys())[0] for i in pheno_list))):
            pheno_type[keys[count]]=pheno_list[count_list][value]
            count=count + 1
            count_list=count_list + 1
        else:
            pheno_type[keys[count]]=[int(x) for x in list(value)]
            count=count + 1
    return pheno_type


def run_model(img_shape, params):
    print(f'In runmodel')
    model1=custom_unet(
        img_shape,
        filters=params["number_filter"],
        kernel=params["kernel"],
        network_depth=params["network_depth"],
        optimiser=params["optimiser"],
        pooling_type=params["pooling_type"],
        activation=params["activation"],
        keep_prob=params["keep_prob"],
        block_size=params["block_size"])
    return model1



input_size=(desired_size,desired_size,3)



# ========= Load settings from Config file
config=configparser.RawConfigParser()
config.read('configuration.txt')
name_experiment=config.get('experiment name', 'name')
# training settings
NGEN=int(config.get('ga settings', 'generations'))
NPOP=int(config.get('ga settings', 'individuals'))
CXPB=float(config.get('ga settings', 'cxpb'))
MUTPB=float(config.get('ga settings', 'mutpb'))
PREV_GEN=None
prev_gen_string=config.get('ga settings', 'previous_population')
use_old_gen=False
if prev_gen_string:
    PREV_GEN=ast.literal_eval(prev_gen_string)
    use_old_gen=True

print("PREV_GEN=", PREV_GEN)


# ====================the GA Code is being kept in here========================
compare=lambda x, y: collections.Counter(x) == collections.Counter(y)

creator.create('Fitness', base.Fitness, weights=(1.0, -1.0))
creator.create('Individual', list, typecode="I", fitness= creator.Fitness, strategy=None)
creator.create("Strategy", list, typecode="I")


INDIVIDUAL_SIZE = sum(design.values())

toolbox = base.Toolbox()

prev_counter = 0
ga_number = 0

def generateES(ind_cls, strg_cls, size):
    print(f'In to_generate')
    global prev_counter
    if not use_old_gen:
        ind = ind_cls(random.randint(0,1) for _ in range(size))
    else:
        print('counter=', prev_counter)
        if prev_counter >= NPOP:
            prev_counter = prev_counter - 20
        ind = ind_cls(PREV_GEN[prev_counter])
        prev_counter += 1

    ind.strategy = strg_cls(random.randint(0,1) for _ in range(size))
    return ind

# generation functions
toolbox.register("individual", generateES, creator.Individual, creator.Strategy, INDIVIDUAL_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# ====================Eval model========================

def eval_model_loss_function(individual,totalParams):
    print(f'In evals')
    chromosome=to_genes(individual)
    params=to_phenos(chromosome, design, pheno_list)
    print(f'GA------------Individual = {individual}, parameters={params}')
    model1=run_model(input_size, params)
    history=model1.fit(x_train, y_train, epochs=10, batch_size=2, verbose=2, validation_data=(x_validate, y_validate), shuffle=True)
    fitness1=max(history.history['val_accuracy'])
    fitness2=min(totalParams)
    fitness_vector=history.history['val_accuracy']
    print(f'GA------------fitness vector={fitness_vector}')
    print(f'---------------------------------------------------------------------')
    print(f'---------------------------------------------------------------------')
    del history
    del model1
    gc.collect()
    return fitness1,fitness2

toolbox.register("evaluate", eval_model_loss_function)

# initialize parameters
pop=toolbox.population(n=NPOP)
hof=tools.HallOfFame(1,similar=np.array_equal)
stats=tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.minimum)
stats.register("max", np.maximum)
stats.register("std", np.std)

logbook = tools.Logbook()
logbook.header = "gen", "evals", "avg", "min", "max", "std"

pop, logbook= algorithms.eaSimple(pop, toolbox, cxpb=CXPB,  mutpb=MUTPB,
                              ngen=NGEN, stats=None, halloffame=hof,
                              verbose=True)

print('GA------------\n', logbook)
print('GA------------Best possible candidates of last generation')
print("Best Gen : ", hof.items[0])

