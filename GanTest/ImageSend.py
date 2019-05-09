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
import socket
import cv2

model = load_model('gan_generator_50.h5', compile=False)
print('Load-Model')
z_dim = 50
#save_dir = 'Predicts'
noise = np.random.normal(0, 1, (1, z_dim))

#fig, ax = plt.subplots()
#count = 0
#epoch = 0
model._make_predict_function()

#------------------------------------------------------
'''
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 6000))
s.listen(1)
print("接続待機中")
soc, addr = s.accept()
print(str(addr)+"と接続完了")
'''
udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
to_send_addr = ('127.0.0.1', 9999)

#------------------------------------------------------

gen_imgs = model.predict(noise)
gen_imgs = 0.5 * gen_imgs + 0.5
while True:
    msg = input("> ")
    img = gen_imgs[0].reshape(28,28)
    print(img)
    #plt.imshow(img, 'gray')
    #plt.show()
    udp.sendto(img.tostring(), to_send_addr)
udp.close()


'''
while True:
    gen_imgs = model.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5


    byteData = gen_imgs.tostring()
    soc.send(byteData)

    noise[0][1] += 0.05
    noise[0][2] += 0.05
    noise[0][3] += 0.05
    noise[0][4] += 0.05
    noise[0][5] += 0.05
    noise[0][6] += 0.05
    noise[0][7] += 0.05
    noise[0][8] += 0.05
    noise[0][9] += 0.05
    count = count + 1
    if count > 50:
        count = count % 50
        noise[0][1] = -1.0
        epoch += 1

    if epoch == 3:
        break
    print(str(count) + 'noise[0][1]' + str(noise[0][1]))

'''