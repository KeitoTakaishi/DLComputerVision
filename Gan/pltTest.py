import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

for loop in range(100):
    data_1d = np.random.rand(22500)
    print(data_1d)
    data_2d = np.reshape(data_1d, (150,150))

    plt.imshow(data_2d, extent=(0,150,0,150),cmap=cm.gist_rainbow)
    plt.pause(1)
    plt.savefig("foo.png")
    #time.sleep(1)
