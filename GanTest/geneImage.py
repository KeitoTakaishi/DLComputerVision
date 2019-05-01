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


model = load_model('/Users/takaishikeito/Documents/ComputerVision/GanTest/gan_generator.h5', compile=False)
print('Load-Model')
z_dim = 100

#gen_imgs = []

#fig, axs = plt.subplots(r, c)
save_dir = '/Users/takaishikeito/Documents/ComputerVision/GanTest/Predicts'
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

noise = np.random.normal(0, 1, (1, z_dim))
print(str(noise.shape))
#noise[0][99] = 0.0

fig, ax = plt.subplots()
count = 0

while (count < 30):
     gen_imgs = model.predict(noise)
     gen_imgs = 0.5 * gen_imgs + 0.5

     ax.imshow(gen_imgs[0].reshape(28, 28), 'gray')
     plt.savefig(os.path.join(save_dir, '{}.png'.format(count)))
     plt.pause(1)

     noise[0][1] += 1.0


     count = count + 1
     print('increment')
