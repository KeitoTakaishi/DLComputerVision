import socket
import numpy as np
import cv2
import cv2
from matplotlib import pyplot as plt

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.bind(('127.0.0.1', 9999))

# 画像を取り続ける
while True:
    stop = input("> ")
    #recive(udp)
    recive_data = bytes()
    buff = 32 * 32 * 16
    jpg_str, add = udp.recvfrom(buff)
    break

print(jpg_str)
narray = np.fromstring(jpg_str, dtype=np.uint)
print(narray)
print(narray.reshape(32,32,3).shape)
img = narray.reshape(32,32,3)
#img = cv2.imdecode(narray, 1)
plt.imshow(img)
plt.show()

udp.close()
