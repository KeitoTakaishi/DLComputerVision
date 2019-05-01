# -*- coding: utf-8 -*-
'''
5x5の画像出力で潜在変数は100次元
Discriminatorは単体で学習を行うが、Generatorは
Discriminatorと合同で学習を行う。
1.  Discriminatorの学習時の入力としては
    28*28のmnist(label=1)とGeneratorによって生成された
    偽画像(label=0)

2.  Combの学習(Generator)の入力は(BatchSize*latestSize)の
    noise.
-------------------------------
#疑問点
- compile自体の意味があやふや
'''
from __future__ import print_function, division

from keras.datasets import mnist
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
import numpy as np

class GAN():
    def __init__(self):
        #mnistデータ用の入力データサイズ
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # 潜在変数の次元数
        self.z_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Discriminatorモデル(見破り)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Generatorモデル(生成)
        self.generator = self.build_generator()
        # generatorは単体で学習しないのでコンパイルは必要ない
        #self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        #連結
        self.combined = self.build_combined1()
        #self.combined = self.build_combined2()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        model = Sequential()
        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        print('discriminator model')
        model.summary()
        return model

    def build_generator(self):

        noise_shape = (self.z_dim,)
        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        print('generator model')
        model.summary()

        return model


    def build_combined1(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model

    def build_combined2(self):
        z = Input(shape=(self.z_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        model = Model(z, valid)
        model.summary()
        return model

    #epoch=30000
    #epochs回分だけ学習データを使い切る
    #一度に使うデータはbatch_sizeとなるので128
    def train(self, epochs=50000, batch_size=128, save_interval=10):

        print('Load Start')
        # mnistデータの読み込み
        (train_images, train_labels),(test_images, test_labels) = mnist.load_data()
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

#if __name__ == '__main__':
gan = GAN()
gan.train()
gan.generator.save('gan_generator_5x5_100.h5')
#gan.combined.save('/Users/takaishikeito/Documents/ComputerVision/Gan/gan_conb.hdf5')
print('main')
