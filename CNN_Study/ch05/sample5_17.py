# -*- coding: utf-8 -*-
'''
学習済みのVGG16モデルの畳み込み層を用いることによって過学習を防ぎ、正解率を向上させる
畳み込みモデルの出力であるconv_base.predictで特徴マップを生成し、
そのデータを全結合分類器のインプットとして使用する。

高速ではあるがデータ数が少ないので過学習に早い段階で陥ってしまう
'''
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16

import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

conv_base = VGG16(weights='imagenet', include_top=False,input_shape=(150,150,3))
conv_base.summary()
#--------------------------------------------------------------------------
#Path-Setting
base_dir = '/Users/takaishikeito/Documents/DLDatasets/cats_and_dog_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
#--------------------------------------------------------------------------
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20


def extract_features(directory, sample_count):
    #VGG16が最終的に出力する特徴マップは(4,4,512) なのでこのモデルが学習するinput形状は(4, 4, 512)
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))#1次元ベクトルの正解ラベル
    generator = datagen.flow_from_directory(directory,
    target_size=(150, 150), batch_size=batch_size,class_mode='binary')

    i = 0
    for inputs_batch, labels_batch in generator:
        feature_batch = conv_base.predict(inputs_batch)
        #VGG16が推論した特徴マップ=このモデルのinput
        features[i * batch_size : (i + 1) * batch_size] = feature_batch
        #正解ラベル
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break

    return feature, labels


train_feature, train_labels = extract_features(train_dir, 2000)
validation_feature, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)
#全結合器に入力出きるように平滑化を行う
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
#--------------------------------------------------------------------------
#model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim= 4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss='binary_crossentropy',metrics=['acc'])

history = model.fit(train_feature, train_labels, epochs=30, batch_size=20,
validation_data=(validation_feature,validation_labels))

#--------------------------------------------------------------------------
#plt
acc = history.history['acc']
val = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training loss')
plt.plot(epochsm val_acc, 'b', label='Validation acc')
plt.title('Traininga and Validation loss')
plt.show()
