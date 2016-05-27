from RunTinyYOLO import *
from custom_loss import get_custom_loss
from data_generator import generate_batch_data
import time
import theano
    
def main():
    # weightFile = '/home/lyf/develop/traffic_light/backup/yolo-tl_82000.weights'
    weightFile = '/home/lyf/develop/darknet/extraction.conv.weights'
    yoloNet = ReadDarkNetWeights(weightFile, 25)
    #reshape weights in every layer
    for i in range(25): #yoloNet.layer_number):
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
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    custom_loss = get_custom_loss(16, 0.5, 1, 1, 5)
    model.compile(optimizer=sgd, loss=custom_loss)
    
    folder = '/home/lyf/develop/traffic_light/crops448/'
    img_list_file = '/home/lyf/develop/traffic_light/croplabel448_train_list.txt'
    model.fit_generator(generate_batch_data(folder, img_list_file, 16), 
    	samples_per_epoch=140000, nb_epoch=50)


if __name__ == '__main__':
    main()