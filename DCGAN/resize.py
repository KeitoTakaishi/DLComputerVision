import cv2
import os
train_images = []
basePath = '../Shutil/faceImages/'
filename = os.listdir(basePath)
width,height=128,128
for i, f in enumerate(filename):
    im = cv2.imread(basePath + f)
    im = cv2.resize(im,(width, height))
    cv2.imwrite( 'Data/' + str(i) + '.jpg', im)

print('Done Resize')



