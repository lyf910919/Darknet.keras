import numpy as np


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

def fill_truth_detection(path, classes, num_boxes):
    truth = [0] * num_boxes * num_boxes * (5 + classes)
    
    labelpath = path[:path.rfind('.jpg')] + '.txt'
    boxes = read_boxes(labelpath)

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