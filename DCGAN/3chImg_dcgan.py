# coding: utf-8

'''
optimizer = Adam(0.0002, 0.5)
dropout
BatchNormalization

'''
from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import numpy as np
import cv2

class DCGAN():
    def __init__(self):
        #input img is 250*250*3
        self.img_rows = 128
        self.img_cols = 128
        self.img_channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.z_dim = 100
        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.generator = self.build_generator()

        self.combined = self.build_combined1()
        #self.combined = self.build_combined2()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):
        noise_shape = (self.z_dim,)

        model = Sequential()

        model.add(Dense(128 * 32 * 32, activation="relu", input_shape=noise_shape))
        model.add(Reshape((32, 32, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(3, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return model


    def build_discriminator(self):
        img_shape = (self.img_rows, self.img_rows, self.img_channels)
        model = Sequential()
        #model.add(Conv2D(64,(5,5), subsample=(2, 2), border_mode='same', img_shape=img_shape))
        model.add(Convolution2D(64,5,5, subsample=(2,2),\
                  border_mode='same', input_shape=img_shape))
        model.add(LeakyReLU(0.2))
        model.add(Convolution2D(128, 5, 5, subsample=(2,2)))
        model.add(LeakyReLU(0.2))
        #fullyConnectedLayer
        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model

    def build_combined1(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model

    def train(self, epochs=10001, batch_size=128, save_interval=100):
        print('Load Start')
        # mnistデータの読み込み
        train_images = []
        basePath = './Data/'
        filename = os.listdir(basePath)
        
        
        for f in filename:
            im = cv2.imread(basePath + f)
            im = im.astype('float32')
            train_images.append(im)
        
        
        
        train_images = np.array(train_images)
        
        print(train_images.shape)
        # 値を-1 to 1に規格化=============================
        train_images = (train_images.astype(np.float32) - 127.5) / 127.5
        
        #train_images = train_images.astype('float32')
        #train_images = train_images/127.5
        #train_images -= np.ones((train_images.shape))
        # ==============================================
        print('Load Done')
        
        
        #axisで指定した位置に1を代入する
        #train_images = np.expand_dims(train_images, axis=3)
        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = self.generator.predict(noise)
            #TruthImg
            idx = np.random.randint(0, train_images.shape[0], half_batch)
            imgs = train_images[idx]

            # Discriminatorを学習
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            #  Generator
            noise = np.random.normal(0, 1, (batch_size, self.z_dim))
            # batch_size行数分[1]ができる
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # 進捗の表示
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % save_interval == 0:
                self.sample_images(epoch)


    def sample_images(self, epoch):
        noise = np.random.normal(0, 1, (1, self.z_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5

        save_dir = './FaceImages'
        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)
        
        print(gen_imgs.shape)
        cv2.imwrite(save_dir+'/{epoch}.jpg'.format(epoch=epoch), gen_imgs[0]*255.0)
        #plt.close()



dcgan = DCGAN()
print(dcgan.z_dim)
dcgan.train()
modelSavingPath = './faceModel.h5' 
dcgan.generator.save(modelSavingPath)
print('Done')
