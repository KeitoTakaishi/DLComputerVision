# coding: utf-8
'''
BatchNormalization
UpSampling2D→同じ値の特徴を使ってフィルターのサイズを大きくする
Convolution2Dが使えない


#画像の補間について
学習し終わった潜在変数を保存，その間での線形補間を行う
#転置畳み込み
特徴フィルタから元画像を復元することが目的
画像サイズを2倍にしていく．
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


class DCGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
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
        model.add(Dense(1024, input_shape=noise_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Reshape((7,7,128), input_shape=(128*7*7,)))
        model.add(UpSampling2D((2,2)))#転置畳み込みのために特徴マップを拡大
        model.add(Convolution2D(64,5,5,border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Convolution2D(1,5,5,border_mode='same'))
        model.add(Activation('tanh'))
        model.summary()

        return model


    def build_discriminator(self):
        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()
        model.add(Convolution2D(64,5,5, subsample=(2,2),\
                  border_mode='same', input_shape=img_shape))
        model.add(LeakyReLU(0.2))
        model.add(Convolution2D(128,5,5,subsample=(2,2)))
        model.add(LeakyReLU(0.2))
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

    def train(self, epochs=30001, batch_size=128, save_interval=100):

        print('Load Start')
        # mnistデータの読み込み
        (train_images, train_labels),(test_images, test_labels) = mnist.load_data()
        print(type(train_images))
        print(train_images.shape)
        print('Load Done')

        # 値を-1 to 1に規格化=============================
        train_images = train_images.astype('float32')
        train_images = train_images/127.5
        train_images -= np.ones((train_images.shape))
        # ==============================================

        #axisで指定した位置に1を代入する
        train_images = np.expand_dims(train_images, axis=3)
        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Discriminatorの学習
            # ---------------------

            # バッチサイズの半数をGeneratorから生成
            noise = np.random.normal(0, 1, (half_batch, self.z_dim))
            #偽画像
            gen_imgs = self.generator.predict(noise)


            # バッチサイズの半数を教師データからピックアップ
            # np.random.randint(a, b) -> a ~ b-1
            idx = np.random.randint(0, train_images.shape[0], half_batch)
            imgs = train_images[idx]

            # Discriminatorを学習
            # 本物データと偽物データは別々に学習させる
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            # それぞれの損失関数を平均
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Generatorの学習
            # ---------------------
            # 平均:0, 標準偏差:1
            noise = np.random.normal(0, 1, (batch_size, self.z_dim))

            # 生成データの正解ラベルは本物（1）
            # batch_size行数分[1]ができる
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # 進捗の表示
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # 指定した間隔で生成画像を保存
            if epoch % save_interval == 0:
                self.sample_images(epoch)



    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        #行と列に分ける
        fig, axs = plt.subplots(r, c)
        save_dir = 'Images'
        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)

        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1

        fig.savefig(os.path.join(save_dir, '{}.png'.format(epoch)))
        plt.close()


dcgan = DCGAN()
dcgan.train()
print('Done')
