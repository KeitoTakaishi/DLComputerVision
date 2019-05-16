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


model = load_model('gan3ch.h5', compile=False)
print('Load-Model')

z_dim = 100


#表示
count = 0
epoch = 0
noise = np.random.normal(0, 1, (1, z_dim))
print(str(noise.shape))

gen_imgs = model.predict(noise)
gen_imgs = 0.5 * gen_imgs + 0.5
im = gen_imgs[0].reshape(250, 250, 3)
#cv2.imwrite('openCVTest.jpg', im*255.0)
'''
while True:
    gen_imgs = model.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5


    #plt.imshow(gen_imgs[0, :, :, 0], cmap='gray')
    #plt.savefig(os.path.join(save_dir, '{}.png'.format('Gan')))
    #plt.pause(0.01)
    #plt.cla()


    #print(gen_imgs[0].reshape(28, 28).shape)
    #cv2.imwrite('Gan.png', gen_imgs[0].reshape(28, 28))

    
    count = count + 1
    if count > 30:
        count = 0
        noise[0][0] = -1.0
        noise[0][3] = -1.0
        noise[0][5] = -1.0
        noise = np.random.normal(0, 1, (r*c, z_dim))
        epoch += 1
    if epoch > 3:
        break
    

    print('increment')
'''