# -*- coding: utf-8 -*-
'''
sample5-5.pyでは過学習に陥ってしまっているので、訓練データを増やす必要がある。
そのためにImageDataGeneratorを用いてデータ拡張を行う。
ImageGenerator検証用のコード

'''
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

datagen = ImageDataGenerator(rotation_range=40,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode='nearest')

original_dataset_dir = '/Users/takaishikeito/Documents/DLDatasets/dogs-vs-cats/train'
base_dir = '/Users/takaishikeito/Documents/DLDatasets/cats_and_dog_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')

fnames = [os.path.join(train_cats_dir, fname)
                        for fname in os.listdir(train_cats_dir)]
print(len(fnames)) #1000

#変形する画像データを選択する
img_path = fnames[1]


'''
img : <PIL.Image.Image image mode=RGB size=150x150 at 0x10D341DD0>
img_to_arrayで配列データに変更->(150, 150, 3)
reshapeで(1, 150, 150, 3)に変更batch_size=1ってことかな
'''
img = image.load_img(img_path, target_size=(150, 150))
#print('img : ' + str(img))
x = image.img_to_array(img)
#print('img_to_array : ' + str(x))
#print('img_to_array : ' + str(x.shape))
x = x.reshape((1,) + x.shape)
#print(x)
#print(str(x.shape))


i = 0
imageNum = 4
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % imageNum == 0:
        break
plt.show()
