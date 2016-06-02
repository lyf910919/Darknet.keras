import os
import numpy as np

from utils.DarkNet import ReadDarkNetWeights, DarkNet

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape, Dropout
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
from model_loader import load_model_h5, load_DAG_model

class box:
    def __init__(self,classes):
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.class_num = 0
        self.probs = np.zeros((classes,1))

def get_activations(model, layer, X_batch):
    # get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    activations = get_activations([X_batch, 0]) # same result as above
    return activations
    
def get_inputs(model, layer, X):
    get_inputs = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].input])
    inputs = get_inputs([X, 0])
    return inputs

def convert_yolo_detections(predictions,classes=1,num=2,square=True,side=11,w=448,h=448,threshold=0.01,only_objectness=0):
    boxes = []
    probs = np.zeros((side*side*num,classes))
    for i in range(side*side):
        row = i / side
        col = i % side
        for n in range(num):
            index = i*num+n
            p_index = side*side*classes+i*num+n
            scale = predictions[p_index]
            box_index = side*side*(classes+num) + (i*num+n)*4

            new_box = box(classes)
            # print [predictions[box_index+t] for t in xrange(4)]
            # print predictions.min()
            new_box.x = (predictions[box_index + 0] + col) / side * w
            new_box.y = (predictions[box_index + 1] + row) / side * h
            new_box.w = pow(predictions[box_index + 2], 2) * w
            new_box.h = pow(predictions[box_index + 3], 2) * h

            for j in range(classes):
                class_index = i*classes
                prob = scale*predictions[class_index+j]
                if(prob > threshold):
                    new_box.probs[j] = prob
                    # print new_box.x, new_box.y, new_box.w, new_box.h, new_box.probs[0]
                else:
                    new_box.probs[j] = 0
            if(only_objectness):
                new_box.probs[0] = scale

            boxes.append(new_box)
    return boxes

def prob_compare(boxa,boxb):
    if(boxa.probs[boxa.class_num] < boxb.probs[boxb.class_num]):
        return 1
    elif(boxa.probs[boxa.class_num] == boxb.probs[boxb.class_num]):
        return 0
    else:
        return -1

def do_nms_sort(boxes,total,classes=1,thresh=0.5):
    for k in range(classes):
        for box in boxes:
            box.class_num = k
        sorted_boxes = sorted(boxes,cmp=prob_compare)
        for i in range(total):
            if(sorted_boxes[i].probs[k] == 0):
                continue
            boxa = sorted_boxes[i]
            for j in range(i+1,total):
                boxb = sorted_boxes[j]
                if(boxb.probs[k] != 0 and box_iou(boxa,boxb) > thresh):
                    boxb.probs[k] = 0
                    sorted_boxes[j] = boxb
    return sorted_boxes

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1/2;
    l2 = x2 - w2/2;
    if(l1 > l2):
        left = l1
    else:
        left = l2
    r1 = x1 + w1/2;
    r2 = x2 + w2/2;
    if(r1 < r2):
        right = r1
    else:
        right = r2
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 or h < 0):
         return 0;
    area = w*h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w*a.h + b.w*b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b)/box_union(a, b);

def drawRects(src, rects):
    for rect in rects:
        print rect
        cv2.rectangle(src, (rect[0]-rect[2]/2, rect[1]-rect[3]/2), (rect[0]+rect[2]/2, rect[1]+rect[3]/2), (0,255,0), 2)
    cv2.imshow('src', src)
    cv2.waitKey(0)
            
def detect(model, image, thresh = 0.2):
    timer = Timer()
    timer.tic()
    data = np.expand_dims(image, axis=0).transpose((0,3,1,2)) / 255.
    # swap channel
    data = data[:,[2,1,0],:,:]
    # modify
    data = data * 2.0 - 1.0
    # print data
    out = model.predict(data)
    # print out[121:363]
    timer.toc()
    # print ('Total detection time is {:.3f}s ').format(timer.total_time)
    
    # for layer in [0,4,8,11,14,17,21,24,27,30,33,36,39,42,45,48,\
    # 52,55,58,61,64,67,70,73,76,79]:
    #     layer_output = get_activations(model, layer, data)[0]
    #     # if layer == 79:
    #     #     layer_output = get_activations(model, layer, data)[0]
    #     #     print layer_output.shape, layer_output.min(), layer_output.max(), layer_output.mean()
    #     #     cnt = 0
    #     #     for val in layer_output.flatten():
    #     #         print cnt, ':', val
    #     #         raw_input()
    #     #         cnt += 1
    #     if len(layer_output.shape) < 4:
    #         continue
    #     # print sum(layer_output[0,:,0,0])
    #     elif layer_output.shape[1] == 3:
    #         vis_square(layer_output)
    #     else:
    #         vis_square(layer_output.reshape(layer_output.shape[0]*layer_output.shape[1],\
    #             layer_output.shape[2], layer_output.shape[3]))
    
    predictions = out[0]
    # return out
    
    # print predictions[363:700]
    boxes = convert_yolo_detections(predictions, \
        classes=1,num=2,square=True,side=11,w=448,h=448,threshold=0.01,only_objectness=0)
    # print 'boxes length:', len(boxes)
    # print 'max prob:', max([box.probs for box in boxes])
    boxes = do_nms_sort(boxes,len(boxes),thresh=0.1)
    rects = [map(int,[box.x, box.y, box.w, box.h])+[box.probs[0]] for box in boxes if box.probs[0] > thresh]
    # drawRects(image, rects)
    return rects

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
    
def main():
    # model = load_model_h5('weights/weights.hdf5')
    model = load_DAG_model('weights/weights_4_20000.h5')
    # model = loadModel('/home/lyf/develop/traffic_light/backup/yolo-tl_82000.weights')
    test_list_file = '/home/lyf/develop/traffic_light/croplabel448test_list.txt'
    with open(test_list_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    f = open('results/croplabel448test_pred.txt', 'w')
    cnt = 0
    for line in lines:
        print cnt
        cnt += 1
        image = cv2.imread(line, cv2.IMREAD_COLOR)
        rects = detect(model, image, 0.1)
        # print 'rects:', len(rects)
        for rect in rects:
            print >> f, line, rect[4][0], rect[0]-rect[2]/2, rect[1]-rect[3]/2, rect[0]+rect[2]/2, rect[1]+rect[3]/2
    f.close()

if __name__ == '__main__':
    main()