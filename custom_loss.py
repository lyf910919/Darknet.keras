import numpy as np
import theano
import theano.tensor as T

def get_bound(boxes):
    left_top = boxes[:,:,:,:2] - boxes[:,:,:,2:] / 2
    right_bot = boxes[:,:,:,:2] + boxes[:,:,:,2:] / 2
    return T.concatenate([left_top, right_bot], axis=3)
    
def get_max(a, b, index, axis=3):
    return T.max(T.concatenate([a[:,:,:,index], b[:,:,:,index]], axis=axis-1)\
      .reshape((a.shape[0],a.shape[1],a.shape[2],2)), axis=axis-1)
    
def get_min(a, b, index, axis=3):
    return T.min(T.concatenate([a[:,:,:,index], b[:,:,:,index]], axis=axis-1)\
      .reshape((a.shape[0],a.shape[1],a.shape[2],2)), axis=axis-1)

def getOverlap(a, b):
    '''
    Given (batch_size, grid_num, box_num, 4) tensors, return 
    (batch_size, grid_num, box_num) overlap area
    '''
    a, b = get_bound(a), get_bound(b)
    xmin = get_max(a, b, 0)
    xmax = get_min(a, b, 2)
    ymin = get_max(a, b, 1)
    ymax = get_min(a, b, 3)
    xside, yside = xmax-xmin, ymax-ymin
    xside = T.switch(T.gt(xside, 0), xside, 0)
    yside = T.switch(T.gt(yside, 0), yside, 0)
    return xside * yside
    
def getIoU(a, b):
    '''
    Given (batch_size, grid_num, box_num, 4) tensors, return 
    (batch_size, grid_num, box_num) IoU score
    '''
    overlap = getOverlap(a, b)
    union = a[:,:,:,2]*a[:,:,:,3] + b[:,:,:,2]*b[:,:,:,3] - overlap
    return overlap / union
    
def get_box_mask(a, b):
    iou = getIoU(a,b)
    m = iou.max(axis=-1, keepdims=True)
    mask = T.eq(iou, m).reshape((a.shape[0], a.shape[1], a.shape[2], 1))
    return mask

def get_custom_loss(batch_size, noobj_scale, obj_scale, class_scale,
  side = 11, classes = 1, objectness = 1, coords = 4, box_num = 2):
    def custom_loss(y_true,y_pred):
        grid_num = side * side
        label_num = objectness + classes + coords
        
        # reshape y_true
        grid_labels = y_true[:, :grid_num*label_num].reshape((batch_size,grid_num,label_num))
        # objectness
        grid_objectness = grid_labels[:,:,0]
        # print grid_objectness.shape.eval()
        # objectness * classes
        grid_classes = grid_labels[:,:,[0]*classes] * grid_labels[:,:,objectness:objectness+classes]
        # objectness * boxes
        grid_boxes = grid_labels[:,:,[0]*coords] * grid_labels[:,:,objectness+classes:label_num]
        
        # get pred classes
        pred_classes = y_pred[:, :grid_num*classes]
        # get pred objectness
        pred_objectness = y_pred[:, grid_num*classes:grid_num*(classes+box_num)]
        # get pred boxes
        pred_boxes = y_pred[:, grid_num*(classes+box_num):]
        
        # noobj loss
        pred_objectness = pred_objectness.reshape((batch_size, grid_num, box_num*coords))
        grid_objectness_mul = T.tile(grid_objectness, (1,box_num))
        noobj_loss = noobj_scale * T.pow(pred_objectness-grid_objectness_mul, 2).sum(axis=2)
        
        # class loss
        pred_classes = pred_classes.reshape((batch_size, grid_num, classes))
        class_loss = class_scale * T.pow(pred_classes-grid_classes, 2) * grid_classes # only calc obj classes
        class_loss.sum(axis=2)
        
        # box loss
        # get object box index
        box_loss = theano.shared(np.zeros((batch_size,1), dtype=np.float32))
        box_index = (grid_objectness > .5).nonzero()
        for i in xrange(box_index[0].shape):
            true_box = grid_boxes[box_index[0][i]][box_index[1][i]]
            for n in xrange(box_num):
              pred_box = pred_boxes[box_index[0][i]][box_index[1][i]][n*coords:(n+1)*coords]
              iou = getIoU(pred_box, true_box)
        return class_loss
    return custom_loss
    
if __name__ == '__main__':
    # loss = get_custom_loss(16,1,1,1)
    # y_pred = T.matrix()
    # y_true = T.matrix()
    # a = theano.function([y_true, y_pred], [loss(y_true, y_pred)])
    a, b = T.tensor4(), T.tensor4()
    iou = getIoU(a,b)
    m = iou.max(axis=-1, keepdims=True)
    mask = T.eq(iou, m).reshape((a.shape[0], a.shape[1], a.shape[2], 1))
    # a_, b_ = get_bound(a), get_bound(b)
    # xmin = get_max(a_, b_, 0)
    # xmax = get_min(a_, b_, 2)
    # ymin = get_max(a_, b_, 1)
    # ymax = get_min(a_, b_, 3)
    # t = T.concatenate([T.shape_padleft(a[:,:,:,0]), T.shape_padleft(b[:,:,:,0])], axis=3)#.reshape((a.shape[0], a.shape[1], a.shape[2], 2))
    # overlap = getOverlap(a, b)
    # union = a[:,:,:,2]*a[:,:,:,3] + b[:,:,:,2]*b[:,:,:,3] - overlap
    # return overlap / union
    f = theano.function([a,b], [get_box_mask(a,b)], on_unused_input='ignore')
    a_val = np.array([[[[5,5,10,10], [6,6,12,12]], [[0,0,0,0], [0,0,0,0]]], [[[5,5,10,10], [6,6,12,12]], [[0,0,0,0], [0,0,0,0]]]])
    b_val = np.array([[[[10,10,20,20], [10,10,20,20]], [[1,1,2,2], [1,1,2,2]]], [[[10,10,20,20], [10,10,20,20]], [[1,1,2,2], [1,1,2,2]]]])
    print a_val.shape, b_val.shape
    print f(a_val, b_val)