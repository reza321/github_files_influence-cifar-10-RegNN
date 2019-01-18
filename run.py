from my_cifar import CIFARModel, load_cifar
import tensorflow as tf
import time
import os
import numpy as np
import matplotlib.pyplot as plt

num_epochs=50
batch_size=128
input_side=32
input_channels=3
weight_decay=0.001
num_classes=10
initial_learning_rate=0.01
damping=1e-2
keep_probs = [1.0, 1.0]
decay_epochs = [10000, 20000]
conv_patch_size = 3



data=load_cifar()
#datam=data.test.x.reshape(data.test.x.shape[0],32,32,3).astype(np.uint8)

model=CIFARModel(
	initial_learning_rate=initial_learning_rate,
	num_classes=num_classes, 
	input_side=input_side,
	input_channels=input_channels,
	data_sets=data,
	num_epochs=num_epochs,
	batch_size=batch_size,
	decay_epochs=decay_epochs,
	train_dir='output', 
    log_dir='log',
    model_name='cifar_mymodel',
	damping=1e-2)

model.train(
    num_steps=data.train.x.shape[0])

model.train(data.train.x)

b=model.get_all_params()
print(b)



















