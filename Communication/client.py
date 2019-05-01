import socket
import numpy as np
import cv2




soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

soc.connect(('127.0.0.1', 6000))

print("接続完了")

while(1):
    data = soc.recv(921600)

    data = np.fromstring(data,dtype=np.uint8)

    data = np.reshape(data,(480,640,1))

    cv2.imshow("",data);


    k = cv2.waitKey(1)
    #if k== 13 :
        #break

#cv2.destroyAllWindows()
