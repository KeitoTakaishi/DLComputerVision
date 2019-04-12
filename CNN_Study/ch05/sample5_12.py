# -*- coding: utf-8 -*-
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

img_path = fnames[1]

img = image.load_img(img_path, target_size=(150, 150))
print(str(img)) # RGB 150 * 150

#(150, 150, 3)
x = image.img_to_array(img)
#(1, 150, 150, 3)
x = x.reshape((1,) + x.shape)
#print(x)


i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
