# -*- coding: utf-8 -*-
from __future__ import print_function, division
from keras.datasets import mnist
from keras.models import load_model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import matplotlib.animation as animation
import numpy as np
import time
import cv2

#------------------------------------------------
#model load & Grid
model = load_model('gan_generator.h5', compile=False)
print('Load-Model')
r, c = 5, 5
z_dim = 100
save_dir = 'TestImage'
#------------------------------------------------
from socket import socket, AF_INET, SOCK_DGRAM
import struct

HOST = ''
PORT = 5000
s = socket(AF_INET, SOCK_DGRAM)
s.bind((HOST, PORT))
#------------------------------------------------

#表示
count = 0
epoch = 0
noise = np.random.normal(0, 1, (r*c, z_dim))



while True:
    gen_imgs = model.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5

    plt.imshow(gen_imgs[0, :, :, 0], cmap='gray')
    plt.savefig(os.path.join(save_dir, '{}.png'.format('Gan')))
    plt.pause(0.01)
    plt.cla()

    #------------------------------
    msg, address = s.recvfrom(8192)
    val = msg.decode('utf-8')
    val = val.replace('\0', '')
    val = float(val)
    val = round(val, 2)
    print(val)
    #------------------------------

    index = np.random.randint(0, 100)
    noise[0][index] += 2.0 * val
    #noise[0][3] += 0.0
    #noise[0][5] += 0.0


    count = count + 1
    if count > 30:
        count = 0
        noise = np.random.normal(0, 1, (r*c, z_dim))
        epoch += 1
    if epoch > 10:
        break


    print('increment')
