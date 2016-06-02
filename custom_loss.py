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
    
def get_box_mask_iou(a, b):
    '''
    return (batch_size, grid_num, box_num, 1) tensor as mask
    '''
    iou = getIoU(a,b)
    m = iou.max(axis=-1, keepdims=True)
    mask = T.eq(iou, m).reshape((a.shape[0], a.shape[1], a.shape[2], 1))
    return mask
    
def get_box_mask_se(a,b):
    '''
    return (batch_size, grid_num, box_num, 1) tensor as mask
    '''
    se = T.pow(T.pow(a-b, 2).sum(axis=-1), .5)
    sem = se.min(axis=-1, keepdims=True) # find the box with lowest square error
    se_mask = T.eq(se, sem).reshape((a.shape[0], a.shape[1], a.shape[2], 1))
    return se_mask
    
def get_box_mask_final(a, b):
    '''
    input: pred_box and true_box tensor of shape (batch_size, grid_num, box_num, 4)
    output: selected box mask tensor of shape (batch_size, grid_num, box_num, 1)
    '''
    mask_iou = get_box_mask_iou(a, b)
    mask_se = get_box_mask_se(a,b)
    mask_sum = mask_iou.sum(axis=-2, keepdims=True) > 1
    mask_final = mask_iou * (1 - mask_sum) + mask_se * mask_sum
    return mask_final

def get_custom_loss(batch_size, noobj_scale, obj_scale, class_scale, coord_scale,
  side = 11, classes = 1, objectness = 1, coords = 4, box_num = 2):
    def custom_loss(y_true,y_pred):
        grid_num = side * side
        label_num = objectness + classes + coords
        ###################################################################
        # reshape y_true
        grid_labels = y_true[:, :grid_num*label_num].reshape((batch_size,grid_num,label_num))
        # objectness
        grid_objectness = grid_labels[:,:,0].reshape((batch_size, grid_num, 1))
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
        ###################################################################
        
        # noobj loss (batch_size, grid_num)
        pred_objectness = pred_objectness.reshape((batch_size, grid_num, box_num))
        grid_objectness_mul = grid_labels[:,:,[0]*box_num]
        noobj_loss = noobj_scale * T.pow(pred_objectness, 2).sum(axis=2) # pred_objectness should be 0
        
        # class loss (batch_size, grid_num)
        pred_classes = pred_classes.reshape((batch_size, grid_num, classes))
        class_loss = class_scale * T.pow(pred_classes-grid_classes, 2) * grid_objectness # only calc obj classes
        class_loss = class_loss.sum(axis=2)
        
        # box loss (batch_size, grid_num)
        # normalize x and y of box
        grid_boxes = grid_boxes[:,:,range(coords)*box_num].reshape((batch_size, grid_num, box_num, coords))
        grid_boxes_n = T.concatenate([grid_boxes[:,:,:,:2]/side, grid_boxes[:,:,:,2:]], axis=-1) #normalize x,y
        pred_boxes = pred_boxes.reshape((batch_size, grid_num, box_num, coords))
        pred_boxes_n = T.concatenate([pred_boxes[:,:,:,:2]/side, pred_boxes[:,:,:,2:]**2], axis=-1) #normalize x,y, square w,h
        # get box mask
        # mask_iou = get_box_mask_iou(grid_boxes_n, pred_boxes_n)
        mask = get_box_mask_final(grid_boxes_n, pred_boxes_n)
        iou = getIoU(grid_boxes_n, pred_boxes_n)
        # get box loss (batch_size, grid_num)
        grid_boxes = T.concatenate([grid_boxes[:,:,:,:2], T.sqrt(grid_boxes[:,:,:,2:])], axis=-1) # square root true w,h
        box_loss = coord_scale * (T.pow(pred_boxes-grid_boxes, 2) * mask)
        box_loss = box_loss.sum(axis=-1)
        box_loss = box_loss * grid_objectness # only calc grids that contain obj
        box_loss = box_loss.sum(axis=-1)
        
        # obj loss (batch_size, grid_num)
        mask = mask.reshape((mask.shape[0], mask.shape[1], mask.shape[2]*mask.shape[3]))
        # obj_loss = obj_scale * T.pow(pred_objectness-grid_objectness_mul, 2) * mask \
        # - noobj_scale * T.pow(pred_objectness, 2) * mask # delete the noobj loss calc before for obj boxes
        # rescore
        obj_loss = obj_scale * T.pow(pred_objectness-iou, 2) * mask \
        - noobj_scale * T.pow(pred_objectness, 2) * mask # delete the noobj loss calc before for obj boxes
        obj_loss = obj_loss * grid_objectness # only calc grids that contain obj
        obj_loss = obj_loss.sum(axis=2)
        loss = noobj_loss+class_loss+box_loss+obj_loss
        loss = loss.sum() / batch_size # batch normalize the loss
        return loss
        # return [noobj_loss.sum(), class_loss.sum(), box_loss.sum(), obj_loss.sum(), grid_objectness, pred_objectness, loss]
    return custom_loss
    
if __name__ == '__main__':
    custom_loss = get_custom_loss(2, 0.5, 1, 1, 5, side = 2, classes = 1, objectness = 1, coords = 4, box_num = 2)
    y_true, y_pred = T.matrix(), T.matrix()
    loss = custom_loss(y_true, y_pred)
    g = T.grad(loss, y_pred)
    f = theano.function([y_true, y_pred], [loss, g], on_unused_input='ignore')
    
    true_val = [0]*6+[1,0,5,5,10,10]+[0]*24+[1,0,1,1,2,2]+[0]*6
    pred_val = [0,1,0,0,
                0,1,1,1,0,0,0,0,
                1,1,1,1, 3,3,1.72,1.72, 1,1,1.41,1.41, 5,5,2.82,2.82] + [0]*16 +\
               [0,1,0,0,
               0,0,0,0,0,0,1,0] + \
               [0]*16+[1,1,1.41,1.41,1,1,1,1]+[0]*8
    true_val = np.array(true_val).reshape((2,24))
    pred_val = np.array(pred_val).reshape((2,44))
    print true_val.shape, pred_val.shape
    for r in f(true_val, pred_val):
      print r