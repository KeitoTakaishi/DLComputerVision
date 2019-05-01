# -*- coding: utf-8 -*-
'''
CNNの中間層の学習を可視化する
'''


img_path = '/Users/takaishikeito/Documents/DLDatasets/cats_and_dog_small/test/cats/cat.1700.jpg'

from keras.preprocessing import image
from keras import models
from keras.applications import VGG16
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from keras.models import load_model
from keras.utils import plot_model


model = load_model('/Users/takaishikeito/Documents/ComputerVision/CNN_Study/ch05/cats_and_dogs_small_2.h5')
#print('########Load-Model-Info###########')
#model.summary()

img = image.load_img(img_path, target_size=(150, 150))
#ndarrayへ変換
img_tensor = image.img_to_array(img) #img_tensor(150, 150, 3)

'''
expand_dimsは第2引数で指定した場所の直前にdim=1 を挿入を挿入する
img_tensor shape(150, 150, 3)
expand_dims shape(1, 150, 150, 3)
'''
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

#-----------------------------------------------------------------
#model
#８つの出力を作る　
layer_outputs = [layer.output for layer in model.layers[:8]]
#特定の入力をもとに。これらの出力を返すモデルを作成
activation_model = models.Model(input=model.input, output=layer_outputs)
#print('########activation-Model-Info###########')
#activation_model.summary()
plot_model(activation_model, to_file='model.png')

activations = activation_model.predict(img_tensor)
for i, activation in enumerate(activations):
    print('activation{} : '.format(i) + str(activation.shape))


first_layer_activation = activations[0]

#-----------------------------------------------------------------
#plt.imshow(img_tensor[0])
plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis')
plt.show()
