import tensorflow as tf
import numpy as np
import os
import sys
import pickle
from six.moves import cPickle
import gzip
import urllib.request
from tensorflow.contrib.learn.python.learn.datasets import base
from dataset import DataSet
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay


def load_batch(fpath):
    f = open(fpath,"rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        # lab = np.identity(10)[arr[0]]
        lab = [arr[0]]        

        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

        labels.append(lab)
        images.append((img/255)-.5)    # To get a good picture comment this and pass only img
        
    return np.array(images),np.array(labels)

def load_cifar():
    train_data = []
    train_labels = []
    
    if not os.path.exists("cifar-10-batches-bin"):
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                   "cifar-data.tar.gz")
        os.popen("tar -xzf cifar-data.tar.gz").read()
        

    for i in range(5):
        r,s = load_batch("cifar-10-batches-bin/data_batch_"+str(i+1)+".bin")
        train_data.extend(r)
        train_labels.extend(s)
        
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    
    test_data, test_labels = load_batch("cifar-10-batches-bin/test_batch.bin")

    test_data=test_data
    train_data=train_data
    # print(train_data.shape)
    # plt.imshow(test_data[30])
    # plt.show()
    # test_data=test_data.reshape(test_data.shape[0],-1)
    # print(train_data[30].shape)
    # print((train_data[30].dtype))
    # test_data=test_data.reshape(test_data.shape[0],32,32,3).astype(np.uint8)
    # plt.imshow(test_data[30])
    # plt.show()
    # exit()
    VALIDATION_SIZE = 5000
    validation_data = train_data[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]

    train_data = train_data[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    train = DataSet(train_data, train_labels)
    validation = DataSet(validation_data, validation_labels)
    test = DataSet(test_data, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)


class CIFARModel(GenericNeuralNet):
    def __init__(self,num_epochs,input_channels,input_side,**kwargs):

        self.input_channels = input_channels
        self.input_side = input_side
        self.input_dim = self.input_side * self.input_side * self.input_channels


        super(CIFARModel, self).__init__(**kwargs)

    def inference(self,input_x):
        # input_x_reshaped=np.reshape(input_x,(input_x.shape[0],self.input_side,self.input_side,self.input_channels))
        input_x_reshaped=tf.reshape(input_x, [-1, self.input_side, self.input_side, self.input_channels])
        conv1=tf.layers.conv2d(inputs=input_x_reshaped,filters=64,kernel_size=3,activation=tf.nn.relu,padding='valid',name='conv1')
        conv2=tf.layers.conv2d(inputs=conv1,filters=64,kernel_size=3,activation=tf.nn.relu,padding='valid',name='conv2')
        pool1=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=1,name='pool1')

        conv3=tf.layers.conv2d(inputs=pool1,filters=128,kernel_size=3,activation=tf.nn.relu,padding='valid',name='conv3')
        conv4=tf.layers.conv2d(inputs=conv3,filters=128,kernel_size=3,activation=tf.nn.relu,padding='valid',name='conv4')
        pool2=tf.layers.max_pooling2d(inputs=conv4,pool_size=[2,2],strides=1,name='pool2')     

        flat1=tf.layers.flatten(pool2)
        dense1=tf.layers.dense(inputs=flat1,units=256,activation=tf.nn.relu,name='dense1')
        dropout=tf.layers.dropout(inputs=dense1,rate=0.5)
        dense2=tf.layers.dense(inputs=dropout,units=256,activation=tf.nn.relu,name='dense2')
        logits=tf.layers.dense(inputs=dense2,units=10,name='logits')
        return logits

    # def inference(self,input_x):
    #     model = Sequential()
    #     input_x_reshaped=np.reshape(input_x,(input_x.shape[0],self.image_size,self.image_size,self.num_channels))            
    #     #input_validation_reshaped=np.reshape(self.data.validation.x,(self.data.validation.x.shape[0],32,32,3))

    #     model.add(Conv2D(64, kernel_size=3,activation='relu',input_shape=input_x_reshaped.shape[1:],name="conv1"))
        
    #     model.add(Conv2D(64, kernel_size=3,activation ='relu',name="conv2"))        
    #     model.add(MaxPooling2D(pool_size=(2, 2)))        

    #     model.add(Conv2D(128, kernel_size=3,activation='relu',name="conv3"))        
    #     model.add(Conv2D(128, kernel_size=3,activation='relu',name="conv4"))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
        
    #     model.add(Flatten())
    #     model.add(Dense(256,activation='relu',name="dense1"))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(256,activation="relu",name="dense2"))
    #     model.add(Dense(10,name="dense3"))
    #     return model

    # def train(self,input_x,num_epochs=50, batch_size=128, train_temp=1):

    #         self.input_x=input_x

    #         self.input_validation_reshaped=np.reshape(self.data.validation.x,(self.data.validation.x.shape[0],self.image_size,self.image_size,self.num_channels))    	

    #         if not os.path.exists(self.file_name):

    #             print("training phase of model ...")

    #             def fn(correct, predicted):
    #                 return tf.nn.softmax_cross_entropy_with_logits(labels=correct,logits=predicted/train_temp)
    #             sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    #             model=self.inference(self.input_x)

    #             model.compile(loss=fn,optimizer=sgd,metrics=['accuracy'])

    #             model.fit(input_x, self.data.train.labels,   
    #                   batch_size=self.batch_size,
    #                 validation_data=(input_validation_reshaped, self.data.validation.labels),
    #                 nb_epoch=self.num_epochs,
    #                 shuffle=True)
                
    #             model.save(self.file_name)        
    #             self.model=model    
    #         else:
    #             print("loading model from hard disk ...")
    #             loaded_model=self.load_trained_model(self.input_validation_reshaped)
    #             self.model=loaded_model
    #         return self.model


    def load_trained_model(self, input_x):
            model = Sequential()           
            
            model=self.inference(input_x)

            model.load_weights( self.file_name)
            return model            

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds 


    def retrain(self, num_steps, feed_dict):        

        retrain_dataset = DataSet(feed_dict[self.input_placeholder], feed_dict[self.labels_placeholder])
        
        for step in xrange(num_steps):   
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)


    def get_all_params(self):
        # names=[n.name for n in tf.get_default_graph().as_graph_def().node]
        all_params = []
        for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'dense1', 'dense2','logits']:        
            for var_name in ['kernel', 'bias']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                all_params.append(temp_tensor)                
        return all_params 


    # def get_all_params_sample(self):
    #     names = [weight.name for layer in self.model.layers for weight in layer.weights]
    #     weights = self.model.get_weights()
    #     for name, weight in zip(names, weights):
    #         print(name, weight.shape)


    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder






