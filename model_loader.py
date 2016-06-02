import os
import numpy as np

from utils.DarkNet import ReadDarkNetWeights, DarkNet

from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape, Dropout
from keras.layers import merge
from keras import backend as K

from math import pow
import theano
import theano.tensor as T

from os import listdir
from os.path import isfile, join
from utils.timer import Timer

import cv2
from matplotlib import pyplot as plt

from custom_loss import get_custom_loss
from data_generator import fill_truth

def SimpleNet(yoloNet):
    model = Sequential()

    #Convolution Layer 2 & Max Pooling Layer 3
    # model.add(ZeroPadding2D(padding=(1,1),input_shape=(3,448,448)))
    l = yoloNet.layers[1]
    if l.weights is None or l.biases is None:
        model.add(Convolution2D(64, 7, 7, input_shape=(3,448,448), init='he_uniform', border_mode='same',subsample=(2,2)))
    else:
        model.add(Convolution2D(64, 7, 7, input_shape=(3,448,448), weights=[yoloNet.layers[1].weights,yoloNet.layers[1].biases],border_mode='same',subsample=(2,2)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    layer_cnt = 3
    
    #Use a for loop to replace all manually defined layers
    for i in range(3,yoloNet.layer_number):
        l = yoloNet.layers[i]
        # print i, len(model.layers)
        if(l.type == "CONVOLUTIONAL"):
            # print l.size, l.n, l.c, l.h, l.w
            sub = (1,1)
            if i == 26: # modify convolution stride
                sub = (2,2)
            model.add(ZeroPadding2D(padding=(l.size//2,l.size//2,)))
            if l.weights is None or l.biases is None:
                model.add(Convolution2D(l.n, l.size, l.size, init='he_uniform', border_mode='valid',subsample=sub))
            else:
                model.add(Convolution2D(l.n, l.size, l.size, weights=[l.weights,l.biases],border_mode='valid',subsample=sub))
            model.add(LeakyReLU(alpha=0.1))
            layer_cnt += 3
        elif(l.type == "MAXPOOL"):
            model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
            layer_cnt += 1
        elif(l.type == "FLATTEN"):
            model.add(Flatten())
            layer_cnt += 1
        elif(l.type == "CONNECTED"):
            # print l.input_size, l.output_size, l.weights.shape, l.biases.shape
            if l.weights is None or l.biases is None:
                model.add(Dense(l.output_size, init='he_uniform'))
            else:
                model.add(Dense(l.output_size, weights=[l.weights,l.biases]))
            layer_cnt += 1
        elif(l.type == "LEAKY"):
            model.add(LeakyReLU(alpha=0.1))
            layer_cnt += 1
        elif(l.type == "DROPOUT"):
            model.add(Dropout(0.5))
            layer_cnt += 1
        else:
            print "Error: Unknown Layer Type",l.type
    return model
            
def init_model(weightFile, upto):
    yoloNet = ReadDarkNetWeights(weightFile, upto)    # 34, 25
    #reshape weights in every layer
    for i in range(upto):
        l = yoloNet.layers[i]
        if(l.type == 'CONVOLUTIONAL'):
            weight_array = l.weights
            n = weight_array.shape[0]
            weight_array = weight_array.reshape((n//(l.size*l.size),(l.size*l.size)))[:,::-1].reshape((n,))
            weight_array = np.reshape(weight_array,[l.n,l.c,l.size,l.size])
            l.weights = weight_array
        if(l.type == 'CONNECTED'):
            weight_array = l.weights
            # weight_array = np.reshape(weight_array,[l.input_size,l.output_size])
            weight_array = np.reshape(weight_array,[l.output_size,l.input_size])
            weight_array = weight_array.transpose()
            # print weight_array.shape
            l.weights = weight_array

    model = SimpleNet(yoloNet)
    return model
    
def loadModel(weight_file):
    model = init_model(weight_file, 34)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    timer = Timer()
    timer.tic()
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    # f = open("weights.txt", "w")
    for i in xrange(len(model.layers)):
        print model.layers[i]
        print model.layers[i].input_shape, model.layers[i].output_shape
        weights = model.layers[i].get_weights()
        if not weights is None and len(weights) > 0:
            print weights[0].shape, weights[0].max(), weights[0].min()
            if len(weights) > 1:
                print weights[0].shape, weights[0].max(), weights[0].min()
                print "layer: %d" % (i)
                # w = weights[0].transpose()
                # w = weights[1]
                # print w.shape
                # cnt = 0
                # for val in w.flatten():
                # #     print >> f, val
                #     print 'weights[1]', cnt, ':', val
                #     cnt += 1
                #     raw_input()
                # print model.layers[4].get_weights()[0].shape, model.layers[4].get_weights()[1].shape
                # weights = model.layers[4].get_weights()[0]
                # weights = weights[0]
                # vis_square(weights.reshape((weights.shape[0]*weights.shape[1], weights.shape[2], weights.shape[3])))
    # f.close()
    timer.toc()
    print 'Total compile time is {:.3f}s'.format(timer.total_time)
    return model
    
def load_model_h5(weight_file):
    darknet = DarkNet()
    model = SimpleNet(darknet)
    model.load_weights(weight_file)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    timer = Timer()
    timer.tic()
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    timer.toc()
    print 'Total compile time is {:.3f}s'.format(timer.total_time)
    for i in xrange(len(model.layers)):
        print model.layers[i]
        print model.layers[i].input_shape, model.layers[i].output_shape
        weights = model.layers[i].get_weights()
        if not weights is None and len(weights) > 0:
            print weights[0].shape, weights[0].max(), weights[0].min()
            # if len(weights) > 1:
            #     # print weights[0].shape, weights[0].max(), weights[0].min()
            #     # print "layer: %d" % (i)
            #     # w = weights[0].transpose()
            #     # w = weights[1]
            #     # print w.shape
            #     # cnt = 0
            #     # for val in w.flatten():
            #     # #     print >> f, val
            #     #     print 'weights[1]', cnt, ':', val
            #     #     cnt += 1
            #     #     raw_input()
            #     # print model.layers[4].get_weights()[0].shape, model.layers[4].get_weights()[1].shape
            #     # weights = model.layers[4].get_weights()[0]
            #     weights = weights[0]
            #     vis_square(weights.reshape((weights.shape[0]*weights.shape[1], weights.shape[2], weights.shape[3])))
    return model

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
        and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
        (0, 1), (0, 1))                 # add some space between filters
        + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.axis('off')
    plt.show()
    
def load_DAG_model(weight_file):
    model = None
    if weight_file.endswith('.weights'):
        model = init_model(weight_file, 25)
    elif weight_file.endswith('.h5') or weight_file.endswith('.hdf5'):
        darknet = DarkNet()
        model = SimpleNet(darknet)
    print model.layers[18]
    print model.layers[18].input_shape, model.layers[18].output_shape
    print model.layers[75]
    print model.layers[75].input_shape, model.layers[75].output_shape
    map56 = model.layers[18].output
    map56 = Convolution2D(128,3,3,init='he_uniform',border_mode='same')(map56)
    map56 = LeakyReLU(alpha=0.1)(map56)
    map56 = Convolution2D(32,3,3,init='he_uniform',border_mode='same')(map56)
    map56 = LeakyReLU(alpha=0.1)(map56)
    # map56 = Convolution2D(8,3,3,init='he_uniform',border_mode='same')(map56)
    # map56 = LeakyReLU(alpha=0.1)(map56)
    map56 = Convolution2D(4,3,3,init='he_uniform',border_mode='same')(map56)
    map56 = LeakyReLU(alpha=0.1)(map56)
    map56_flat = Flatten()(map56)
    map7_flat = model.layers[75].output
    concat = merge([map56_flat, map7_flat], mode='concat')
    fc = Dense(4096, init='he_uniform')(concat)
    fc = Dropout(0.5)(fc)
    fc = LeakyReLU(alpha=0.1)(fc)
    fc = Dense(1331, init='he_uniform')(fc)
    new_model = Model(input=model.inputs, output=fc)
    print new_model.layers[-1].output_shape
    print len(new_model.layers)
    if weight_file.endswith('.h5') or weight_file.endswith('.hdf5'):
            new_model.load_weights(weight_file)
    return new_model
    
def main():
    model = load_DAG_model('/home/lyf/develop/darknet/extraction.conv.weights')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    timer = Timer()
    timer.tic()
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    timer.toc()
    print 'model compiled in %s seconds.' % timer.total_time

if __name__ == '__main__':
    main()