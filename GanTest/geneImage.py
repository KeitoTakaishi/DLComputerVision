# -*- coding: utf-8 -*-
'''
1枚の画像を生成
'''
from __future__ import print_function, division

import keras
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

print('------------------------------------')
print(keras.__version__)
#py35 -> 2.2.2
# Gan ->   
print('------------------------------------')
#model = load_model('gan_generator_50.h5', compile=False)
model = load_model('gan_generator_50.h5')
print('Load-Model')
z_dim = 50

#gen_imgs = []

#fig, axs = plt.subplots(r, c)
save_dir = 'Predicts'
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

noise = np.random.normal(0, 1, (4, z_dim))
start = np.random.normal(0, 1, (4, z_dim))
target = np.random.normal(0, 1, (4, z_dim))

vec = target - start
print('noise shape: ' + str(start.shape))
print('vec shape: ' + str(vec.shape))



count = 0
epoch = 0
model._make_predict_function()

r = 2
c = 2
cnt = 0
#fig, axs = plt.subplots(r, c)
while True:
    '''
    gen_imgs = model.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    cnt = 0
    for i in range(r):
        for j in range(c):
            #axs[i,j].imshow(gen_imgs[cnt].reshape(28, 28), 'gray')
            axs[i,j].imshow(gen_imgs[cnt, :, :, 0], 'gray')
            #axs[i,j].axis('off')
            cnt = cnt + 1
    plt.savefig(os.path.join(save_dir, '{}.png'.format('Gan')))
    plt.pause(0.001)
    plt.cla()


    noise[0][0] += 0.1
    noise[0][1] += 0.05
    noise[0][2] += 0.1
    noise[0][3] += 0.05

    noise[1][0] += 0.05
    noise[1][1] += 0.1

    noise[2][0] += 0.05
    noise[2][1] += 0.1

    noise[3][0] += 0.1
    noise[3][1] += 0.05

    count = count + 1
    if count > 50:
        count = count % 50
        gen_imgs = model.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        # noise[0][1] = -3.0
        # noise[1][1] = -3.0
        # noise[2][1] = -3.0
        # noise[3][1] = -3.0
        epoch += 1

    print(epoch)
    if epoch == 1:
        break
    '''

    gen_imgs = model.predict(start)
    gen_imgs = 0.5 * gen_imgs + 0.5
    print('gen_imgs shape:' + str(gen_imgs.shape))
    target = gen_imgs.reshape(4, 28, 28, 1)[0]
    print(target.shape)
    rgb_gen_imgs = cv2.cvtColor(target,cv2.COLOR_GRAY2RGB)
    print('rgb_gen_imgs shape:' + str(rgb_gen_imgs.shape))

    #plt.imshow(gen_imgs[0, :, :, 0], cmap='gray')
    plt.imshow(rgb_gen_imgs, cmap = 'BrBG')

    rgb_gen_imgs = cv2.resize(rgb_gen_imgs, None, fx = 10, fy = 10)
    print(rgb_gen_imgs)
    cv2.imwrite("LennaG.png",rgb_gen_imgs*255)
    #plt.savefig(os.path.join(save_dir, '{}.png'.format('Gan')))
    plt.pause(0.001)
    plt.cla()

    start[0][0] += vec[0][0] * 0.1

    count = count + 1
    if count > 1:
        start = np.random.normal(0, 1, (4, z_dim))
        target = np.random.normal(0, 1, (4, z_dim))
        vec = target - start
        epoch += 1
        count = 0
    if epoch == 2:
        break
