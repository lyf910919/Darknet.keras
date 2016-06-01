import os
import numpy as np

from utils.readImgFile import readImg
from utils.TinyYoloNet import ReadTinyYOLONetWeights
from utils.DarkNet import ReadDarkNetWeights, DarkNet
from utils.crop import crop

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape, Dropout
from keras import backend as K

from math import pow
import theano
import theano.tensor as T

from PIL import Image
from PIL import ImageDraw

from os import listdir
from os.path import isfile, join
from utils.timer import Timer

import cv2
from matplotlib import pyplot as plt

from custom_loss import get_custom_loss
from data_generator import fill_truth

class box:
    def __init__(self,classes):
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.class_num = 0
        self.probs = np.zeros((classes,1))

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

def draw_detections(impath,num,thresh,boxes,classes,labels,save_name):
    """
    Args:
        impath: The image path
        num: total number of bounding boxes
        thresh: boxes prob beyond this thresh will be drawn
        boxes: boxes predicted by the network
        classes: class numbers of the objects
    """
    img = Image.open(impath)
    drawable = ImageDraw.Draw(img)
    ImageSize = img.size
    for i in range(num):
        #for each box, find the class with maximum prob
        max_class = np.argmax(boxes[i].probs)
        prob = boxes[i].probs[max_class]
        if(prob > thresh):
            b = boxes[i]

            temp = b.w
            b.w = b.h
            b.h = temp

            left  = (b.x-b.w/2.)*ImageSize[0];
            right = (b.x+b.w/2.)*ImageSize[0];
            top   = (b.y-b.h/2.)*ImageSize[1];
            bot   = (b.y+b.h/2.)*ImageSize[1];

            if(left < 0): left = 0;
            if(right > ImageSize[0]-1): right = ImageSize[0]-1;
            if(top < 0): top = 0;
            if(bot > ImageSize[1]-1): bot = ImageSize[1]-1;

            print "The four cords are: ",left,right,top,bot
            drawable.rectangle([left,top,right,bot],outline="red")
            img.save(os.path.join(os.getcwd(),'results',save_name))
            print labels[max_class],": ",boxes[i].probs[max_class]
            
def drawRects(src, rects):
    for rect in rects:
        print rect
        cv2.rectangle(src, (rect[0]-rect[2]/2, rect[1]-rect[3]/2), (rect[0]+rect[2]/2, rect[1]+rect[3]/2), (0,255,0), 2)
    cv2.imshow('src', src)
    cv2.waitKey(0)
            
def loadModel(weightFile):
    yoloNet = ReadDarkNetWeights(weightFile, 34)#25)
    #reshape weights in every layer
    for i in range(yoloNet.layer_number):
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
            print weight_array.shape
            l.weights = weight_array

    model = SimpleNet(yoloNet)

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
    
def detect(model, image):
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
    print ('Total detection time is {:.3f}s ').format(timer.total_time)
    
    # for layer in []:
    #     shape = model.layers[layer].input_shape
    #     print shape
    #     if len(shape) == 4:
    #         layer_input = get_inputs(model, layer, data)[0]
    #         print type(layer_input), layer_input.shape
    #         # for b in xrange(1):
    #         for c in xrange(shape[1]):
    #             for h in xrange(shape[2]):
    #                 for w in xrange(shape[3]):
    #                     print '(',c,h,w,'):', layer_input[0][c][h][w]
    #                     raw_input()
    
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
    print 'boxes length:', len(boxes)
    print 'max prob:', max([box.probs for box in boxes])
    boxes = do_nms_sort(boxes,len(boxes))
    rects = [map(int,[box.x, box.y, box.w, box.h])+[box.probs[0]] for box in boxes if box.probs[0] > 0.3]
    # print rects[:4]
    drawRects(image, rects)
    # draw_detections(os.path.join(imagePath,image_name),98,0.2,boxes,20,labels,image_name)
    #draw_detections(os.path.join(os.getcwd(),'resized_images','1.jpg'),98,0.2,boxes,20,labels,image_name)
    # return get_activations(model, 17, data)[0].flatten()

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
    #image = readImg(os.path.join(os.getcwd(),'images/Yolo_dog.img'),h=448,w=448)
    labels = ["traffic_light"]
    model = load_model_h5('weights.hdf5')
    # model = loadModel('/home/lyf/develop/traffic_light/backup/yolo-tl_82000.weights')
    test_list_file = '/home/lyf/develop/traffic_light/croplabel448test_list.txt'
    with open(test_list_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    # batch_size = 1
    # custom_loss = get_custom_loss(batch_size, 0.02, 0.5, 0.1, 1)
    # y_true_t, y_pred_t = T.matrix(), T.matrix()
    # loss = custom_loss(y_true_t, y_pred_t)
    # f = theano.function([y_true_t, y_pred_t], [loss[0], loss[1], loss[2], loss[3], loss[4], loss[5], loss[6]])
    # print 'function compiling done'
    for line in lines:
        image = cv2.imread(line, cv2.IMREAD_COLOR)
        # y_true = fill_truth(line, 1, 11)
        # y_true = np.expand_dims(y_true, axis=0)
        # print 'y_true', y_true.shape
        # image = crop(line, resize_width=512,resize_height=512,new_width=448,new_height=448)
        # image = np.expand_dims(image, axis=0)
        predictions = detect(model, image)
        # print 'predictions', predictions.shape
        # with open('loss_output.txt', 'w') as file1:
        #     for l in f(y_true.astype(np.float32), predictions.astype(np.float32)):
        #         print >> file1, l
        # cv2.imshow('image', image)
        # cv2.waitKey(0)

if __name__ == '__main__':
    main()