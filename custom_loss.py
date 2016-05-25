import numpy as np
import theano
import theano.tensor as T

def get_bound(box):
    a = box[:2] - box[2:] / 2
    b = box[:2] + box[2:] / 2
    return theano.T.concatenate([a,b], axis=0)

def getOverlap(a, b):
    if a[0] >= b[2] or b[0] >= a[2] or a[1] >= b[3] or b[1] >= a[3]:
        return 0.0
    xmin = T.max([a[0], b[0]])
    xmax = T.min([a[2], b[2]])
    ymin = T.max([a[1], b[1]])
    ymax = T.min([a[3], b[3]])
    return (xmax-xmin) * (ymax-ymin)
    
def getIoU(a,b):
    a, b = get_bound(a), get_bound(b)
    overlap = getOverlap(a,b)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / union

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
        for i in xrange(box_index[0].shape.eval()):
            true_box = grid_boxes[box_index[0][i]][box_index[1][i]]
            for n in xrange(box_num):
              pred_box = pred_boxes[box_index[0][i]][box_index[1][i]][n*coords:(n+1)*coords]
              iou = getIoU(pred_box, true_box)
        return class_loss
    return custom_loss
    
if __name__ == '__main__':
    loss = get_custom_loss(16,1,1,1)
    y_pred = theano.shared(np.zeros((16,1331)))
    y_true = theano.shared(np.zeros((16,1331)))
    a = theano.function([y_true, y_pred], [loss(y_true, y_pred)])