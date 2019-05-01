# -*- coding: utf-8 -*-
'''
1枚の画像を生成
'''
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


model = load_model('gan_generator_50.h5', compile=False)
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
print(str(noise.shape))
#noise[0][99] = 0.0


count = 0
epoch = 0
model._make_predict_function()

r = 2
c = 2
cnt = 0
fig, axs = plt.subplots(r, c)
while True:

    gen_imgs = model.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5

    cnt = 0
    for i in range(r):
        for j in range(c):
            #axs[i,j].imshow(gen_imgs[cnt].reshape(28, 28), 'gray')
            axs[i,j].imshow(gen_imgs[cnt, :, :, 0], 'gray')
            #axs[i,j].axis('off')
            cnt = cnt + 1
    plt.pause(0.001)
    plt.cla()


    noise[0][0] += 0.1
    noise[0][1] += 0.05

    noise[1][0] += 0.05
    noise[1][1] += 0.1

    noise[2][0] += 0.05
    noise[2][1] += 0.1

    noise[3][0] += 0.1
    noise[3][1] += 0.05

    count = count + 1
    if count > 50:
        count = count % 50
        noise[0][1] = -3.0
        noise[1][1] = -3.0
        noise[2][1] = -3.0
        noise[3][1] = -3.0
        epoch += 1

    print(epoch)
    if epoch == 1:
        break
    '''
    ax.imshow(gen_imgs[0].reshape(28, 28), 'gray')
    #plt.savefig(os.path.join(save_dir, '{}.png'.format(count)))
    plt.pause(0.01)
    plt.cla()


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
