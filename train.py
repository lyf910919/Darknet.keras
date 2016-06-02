from model_loader import load_DAG_model, init_model
from custom_loss import get_custom_loss
from data_generator import generate_batch_data
import time
import theano
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras import backend as K
from keras.optimizers import SGD
    
class LossHistory(Callback):
    def __init__(self, loss_file):
        self.loss_file = loss_file
        assert(os.path.exists(loss_file))
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        if batch % 1000 == 0:
            f = open(self.loss_file, 'w+')
            for loss in self.losses:
                print >> f, loss
            f.close()
            self.losses = []
            
class SaveWeights(Callback):
    def __init__(self, weights_file_prefix):
        self.weights_file_prefix = weights_file_prefix
        self.epoch = 0
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
    def on_batch_end(self, batch, logs={}):
        if batch % 5000 == 0 and batch != 0:
            weights_name = self.weights_file_prefix+'_%s_%s.h5'%(self.epoch, batch)
            print 'save weights: ' + weights_name
            self.model.save_weights(weights_name, overwrite=True)

class LrScheduler(Callback):
    def __init__(self, batch_point, lr):
        self.batch_num = 0
        self.batch_point = batch_point
        self.lr = lr
    def on_train_begin(self, logs={}):
        self.batch_num = 0
    def on_batch_begin(self, batch, logs={}):
        # print self.batch_num
        for i in xrange(len(self.batch_point)):
            if self.batch_num < self.batch_point[i]:
                break
            elif self.batch_num == self.batch_point[i]:
                if i < len(self.lr):
                    K.set_value(self.model.optimizer.lr, self.lr[i])
                print 'current lr:', K.get_value(self.model.optimizer.lr)
        self.batch_num += 1

def schedule(epoch):
    if epoch < 1:
        return 0.000005
    else:
        return 0.0001

def main():
    # weightFile = '/home/lyf/develop/traffic_light/backup/yolo-tl_82000.weights'
    # weightFile = '/home/lyf/develop/darknet/extraction.conv.weights'
    weightFile = 'weights/weights_4_20000.h5'
    model = load_DAG_model(weightFile)
    # model = init_model(weightFile, 25)
    batch_size = 4
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    custom_loss = get_custom_loss(batch_size, noobj_scale=0.01, obj_scale=1, class_scale=0.01, coord_scale=5)
    model.compile(optimizer=sgd, loss=custom_loss)
    print 'compile done'
    
    folder = '/home/lyf/develop/traffic_light/crops448/'
    # img_list_file = '/home/lyf/develop/traffic_light/croplabel448_train_list.txt'
    img_list_file = '/home/lyf/develop/traffic_light/croplabel448_list.txt'
    # f = theano.function([], [])
    # for X, y in generate_batch_data(folder, img_list_file, batch_size):
    #     pred = model.predict(X)[0]
    #     print pred
    #     print pred.max(), pred.min(), pred.mean()
    #     break
    history = LossHistory('loss.txt')
    save = SaveWeights('weights/weights0602')
    checkpointer = ModelCheckpoint(filepath='weights/weights.hdf5', verbose=1)
    # scheduler = LearningRateScheduler(schedule)
    batch_point = [0, 40000, 80000, 120000]#, 40, 80, 120, 200, 5000, 10000, 20000, 40000, 60000, 100000]
    lr = [0.001, 0.0005, 0.0002, 0.0001]#0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.0005, 0.0002, 0.0001]
    scheduler = LrScheduler(batch_point, lr)
    model.fit_generator(generate_batch_data(folder, img_list_file, batch_size), 
    	samples_per_epoch=140000, nb_epoch=50, callbacks=[history, checkpointer, scheduler, save])


if __name__ == '__main__':
    main()