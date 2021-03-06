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


model = load_model('gan_generator.h5', compile=False)
print('Load-Model')
r, c = 5, 5
z_dim = 100

#gen_imgs = []

#fig, axs = plt.subplots(r, c)
save_dir = 'TestImage'
# count = 0
# for i in range(10):
#     #画像の生成
#     img = model.predict(noise)
#     img = 0.5 * img + 0.5
#     gen_imgs.append(img)
#     #plt
#     plt.imshow(gen_imgs[count])
#     plt.savefig(os.path.join(save_dir, '{}.png'.format(count)))
#     count += 1
# plt.close()


#表示
count = 0
epoch = 0
noise = np.random.normal(0, 1, (r*c, z_dim))
print(str(noise.shape))
noise[0][99] = 0.0
while True:
    gen_imgs = model.predict(noise)
    #gen_imgs = 0.5 * gen_imgs + 0.5

    plt.imshow(gen_imgs[0, :, :, 0], cmap='gray')
    #plt.savefig(os.path.join(save_dir, '{}.png'.format('Gan')))
    plt.pause(0.01)
    plt.cla()


    #print(gen_imgs[0].reshape(28, 28).shape)
    #cv2.imwrite('Gan.png', gen_imgs[0].reshape(28, 28))


    noise[0][0] += 0.3
    noise[0][3] += 0.2
    noise[0][5] += 0.2
    '''
    noise[0][0] += 0.5
    noise[1][0] += 0.5
    noise[2][0] += 0.5
    noise[3][0] += 0.5
    noise[4][0] += 0.5
    noise[5][0] += 0.5
    noise[6][0] += 0.5
    noise[7][0] += 0.5
    noise[8][0] += 0.5
    noise[9][0] += 0.5
    noise[10][0] += 0.5
    noise[11][0] += 0.5
    '''

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
