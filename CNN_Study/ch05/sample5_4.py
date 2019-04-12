import os, shutil

original_dataset_dir = '/Users/takaishikeito/Documents/DLDatasets/dogs-vs-cats/train'
base_dir = '/Users/takaishikeito/Documents/DLDatasets/cats_and_dog_small'
if os.path.exists(base_dir) == False:
    os.mkdir(base_dir)


#訓練、テスト、検証用のディレクトリの配置
train_dir = os.path.join(base_dir, 'train')
if os.path.exists(train_dir) == False:
    os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
if os.path.exists(validation_dir) == False:
    os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
if os.path.exists(test_dir) == False:
    os.mkdir(test_dir)

#訓練用の犬猫
train_cats_dir = os.path.join(train_dir, 'cats')
if os.path.exists(train_cats_dir) == False:
    os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
if os.path.exists(train_dogs_dir) == False:
    os.mkdir(train_dogs_dir)

#検証用
validation_cats_dir = os.path.join(validation_dir, 'cats')
if os.path.exists(validation_cats_dir) == False:
    os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if os.path.exists(validation_dogs_dir) == False:
    os.mkdir(validation_dogs_dir)

#test
test_cats_dir = os.path.join(test_dir, 'cats')
if os.path.exists(test_cats_dir) == False:
    os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
if os.path.exists(test_dogs_dir) == False:
    os.mkdir(test_dogs_dir)


#最初の1000枚の猫画像をtrain_cats_dirにコピー
#formatによって{}の中にはiが代入される
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i)  for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copy(src, dst)
print('total training cat images : ' , len(os.listdir(train_cats_dir)))
print('total training dog images : ' , len(os.listdir(train_dogs_dir)))

print('total validation cat images : ', len(os.listdir(validation_cats_dir)))
print('total calidation dog images : ', len(os.listdir(validation_dogs_dir)))

print('total test cat images : ', len(os.listdir(test_cats_dir)))
print('total test dog images : ', len(os.listdir(test_dogs_dir)))
