import numpy as np
import cv2
import random
import time


class Box():
    def __init__(self):
        self.id = 0
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

def read_boxes(labelpath):
    with open(labelpath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    boxes = []
    for line in lines:
        line = line.split()
        assert(len(line) == 5)
        box = Box()
        box.id = int(line[0])
        box.x, box.y, box.w, box.h = map(float, line[1:])
        boxes.append(box)
    return boxes

def fill_truth(path, classes, num_boxes, box_dict = None):
    # can pre-read label files in dict
    truth = [0] * num_boxes * num_boxes * (5 + classes)
    if box_dict is None or len(box_dict) == 0:
        labelpath = path[:path.rfind('.jpg')] + '.txt'
        boxes = read_boxes(labelpath)
    else:
        boxes = box_dict[path]
        
    for box in boxes:
        if box.w < .01 or box.h < .01:
            continue
            
        col = int(box.x*num_boxes)
        row = int(box.y*num_boxes)
        x = box.x*num_boxes - col;
        y = box.y*num_boxes - row;
        w = box.w
        h = box.h
        
        '''
        #################################################################
        #objectness{0,1}, one_hot class label vector[0....0], x, y, w, h#
        #################################################################
        '''
        index = (col+row*num_boxes)*(5+classes)
        if truth[index]:
            continue
        truth[index] = 1
        index += 1
        
        if box.id < classes:
            truth[index+box.id] = 1
        index += classes

        truth[index:index+4] = x, y, w, h
    return np.array(truth)
    
def preprocess(src):
    # make channel the first dimention and scale
    data = src.transpose((2,0,1)) / 255.
    # swap channel
    data = data[[2,1,0],:,:]
    # modify according to darknet
    data = data * 2.0 - 1.0
    return data

def load_data(paths, img_shape=(448,448,3), classes=1, sides=11, box_dict=None):
    imgs, y_trues = [], []
    for path in paths:
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        assert(not src is None)
        assert(src.shape == img_shape)
        img = preprocess(src)
        y_true = fill_truth(path, classes, sides, box_dict)
        imgs.append(img)
        y_trues.append(y_true)
    return imgs, y_trues
    
def generate_batch_data(folder, img_list_file, batch_size, img_shape=(448,448,3), classes=1, sides=11):
    """
    Args:
      folder: the path of image and label folder
      img_list_file: the path of the file of image list
      batch_size: batch size
    Funcs:
      A data generator generates training batch indefinitely
    """
    with open(img_list_file, 'r') as f:
        img_list = [line.strip() for line in f.readlines() if len(line) > 1]
        
    # # label file pre-read
    # box_dict = {}
    # for img in img_list:
    #     labelpath = img[:img.rfind('.jpg')] + '.txt'
    #     boxes = read_boxes(labelpath)
    #     box_dict[img] = boxes
    # print len(box_dict), 'label files pre-read'

    while True:
        random.shuffle(img_list)
        batches = len(img_list) // batch_size
        for b in xrange(batches):
            paths = img_list[b*batch_size:(b+1)*batch_size]
            imgs, y_trues = load_data(paths, img_shape, classes, sides)
            yield np.asarray(imgs), np.asarray(y_trues)
           
def main():
    folder = '/home/lyf/develop/traffic_light/crops448/'
    img_list_file = '/home/lyf/develop/traffic_light/croplabel448_train_list.txt'
    ts, te = time.time(), time.time()
    for X, y_trues in generate_batch_data(folder, img_list_file, 16):
        te = time.time()
        print 'batch loading time:', te-ts
        print X.shape, y_trues.shape
        for i in xrange(len(X)):
            img = X[i]
            y_true = y_trues[i]
            img = (img+1.0)/2.0
            img = img.transpose((1,2,0))[:,:,[2,1,0]]*255
            img = img.astype(np.uint8).copy()
            grid_labels = y_true.reshape((121,6))
            boxes = grid_labels[:,[0,0,0,0]] * grid_labels[:,[2,3,4,5]]
            print boxes.shape
            for i in xrange(boxes.shape[0]):
                row = i / 11
                col = i % 11
                x = (col+boxes[i][0]) / 11 * 448
                y = (row+boxes[i][1]) / 11 * 448
                w = boxes[i][2] * 448
                h = boxes[i][3] * 448
                x,y,w,h = map(int, [x,y,w,h])
                print x,y,w,h
                cv2.rectangle(img, (x-w/2, y-h/2), (x+w/2, y+h/2), (0,255,0), 2)
            cv2.imshow('img', img)
            cv2.waitKey()
        raw_input()
        ts = time.time()
    
if __name__ == '__main__':
    main()