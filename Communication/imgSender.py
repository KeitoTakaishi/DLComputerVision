# conding : utf-8
#client
import socket
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random



img = np.zeros([3, 3], dtype = np.float16)
img[0,1] = 1.0
img[1,1] = 1.0
img[2,1] = 1.0


print(img)

#socketを用意
udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
to_send_addr = ('127.0.0.1', 9999)
while True:
    msg = input("> ")
    udp.sendto(img.tobytes(), to_send_addr)
    #print(img.tobytes())
    #udp.sendto(b'1', to_send_addr)
udp.close()
