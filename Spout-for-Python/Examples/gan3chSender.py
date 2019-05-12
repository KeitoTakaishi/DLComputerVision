# Add relative directory ../Library to import path, so we can import the SpoutSDK.pyd library. Feel free to remove these if you put the SpoutSDK.pyd file in the same directory as the python scripts.
import sys
sys.path.append('../Library')
import os
import argparse
import cv2
import SpoutSDK
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

#------------------------------
#from __future__ import print_function, division
from keras.datasets import mnist
from keras.models import load_model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
#------------------------------

"""parsing and configuration"""
def parse_args():
    desc = "Spout for Python webcam sender example"
    #desc = "SpoutPy"
    parser = argparse.ArgumentParser(description=desc)
   
    parser.add_argument('--camSize', nargs = 2, type=int, default=[640, 480], help='File path of content image (notation in the paper : x)')

    parser.add_argument('--camID', type=int, default=1, help='Webcam Device ID)')

    return parser.parse_args()


"""main"""
def main():

    # parse arguments
    args = parse_args()

    # window details
    width = args.camSize[0] 
    height = args.camSize[1] 
    display = (width,height)
    
    # window setup
    pygame.init() 
    pygame.display.set_caption('Spout for Python Webcam Sender Example')
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    pygame.display.gl_set_attribute(pygame.GL_ALPHA_SIZE, 8)

    # init capture & set 
    '''
    cap = cv2.VideoCapture(args.camID)
    cap.set(3, width)
    cap.set(4, height)
    '''

    # OpenGL init
    glMatrixMode(GL_PROJECTION)
    glOrtho(0,width,height,0,1,-1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glClearColor(0.0,0.0,0.0,0.0)
    glEnable(GL_TEXTURE_2D)
   
    # init spout sender
    spoutSender = SpoutSDK.SpoutSender()
    spoutSenderWidth = width
    spoutSenderHeight = height
    # Its signature in c++ looks like this: bool CreateSender(const char *Sendername, unsigned int width, unsigned int height, DWORD dwFormat = 0);
    #spoutSender.CreateSender('Spout for Python Webcam Sender Example', width, height, 0)
    spoutSender.CreateSender('SpoutPy', width, height, 0)
    
    # create texture id for use with Spout
    senderTextureID = glGenTextures(1)

    # initalise our sender texture
    glBindTexture(GL_TEXTURE_2D, senderTextureID)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, 0)

    #---------------------------------
    #predict
    path = os.getcwd()
    print('------------------')
    modelPath = 'gan-generateFace01.h5'
    model = load_model(modelPath, compile=False)
    z_dim = 100
    model._make_predict_function()

    start = np.random.normal(0, 1, (1, z_dim))
    end = np.random.normal(0, 1, (1, z_dim))
    v = end - start
    gen_imgs = model.predict(start)
    #gen_imgs = 0.5 * gen_imgs + 0.5
    cnt = 0
    #---------------------------------
    from socket import socket, AF_INET, SOCK_DGRAM
    import struct
    
    HOST = ''
    PORT = 7000
    s = socket(AF_INET, SOCK_DGRAM)
    s.bind((HOST, PORT))
    val = 0.0
    

    #---------------------------------
    # loop 
    while(True):
        for event in pygame.event.get():
           if event.type == pygame.QUIT:
               pygame.quit()
               quit()

        #cal-predict
        if cnt > 35:
            start = np.random.normal(0, 1, (1, z_dim))
            end = np.random.normal(0, 1, (1, z_dim))
            v = end - start
            v *= 0.01
            cnt = 0
        
        gen_imgs = model.predict(start)
        gen_imgs = 0.5 * gen_imgs + 0.5
        target = gen_imgs.reshape(250, 250, 3)
        
        #rgb_gen_imgs = target
        #rgb_gen_imgs = rgb_gen_imgs.astype('uint8')
        #rgb_gen_imgs = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        #print('rgb imgs :  '+ str(rgb_gen_imgs.shape)) 

        #cv2.cvtColor(rgb_gen_imgs, target, cv2.COLOR_BGR2RGB)
        
        frame = np.zeros((250, 250, 3), dtype="float16")
        #print('shape : ' + str(frame.shape))
        frame = target
        cv2.imwrite("faceTest.png", frame)
        msg, address = s.recvfrom(8192)
        val = msg.decode()
        val = (float)(val.replace('\0', ''))
        start += val * 0.01
        cnt += 1
        '''
        print('shape : ' + str(frame.shape))
        print('shape : ' + str(frame[:, :, 0].size))
        print('shape : ' + str(frame[:, :, 1].size))
        print('shape : ' + str(frame[:, :, 2].size))
        '''

        #-----------------------------------------------
        
        #camera
        '''
        ret, frame = cap.read()
        print('frame-shape' + str(frame.shape))
        #frame = cv2.imread('../Images/2500.png')
        frame = cv2.flip(frame, 1 )
        print('shape : ' + str(frame.shape))
        print('shape : ' + str(frame[:, :, 0].size))
        print('shape : ' + str(frame[:, :, 1].size))
        print('shape : ' + str(frame[:, :, 2].size))
        '''
        #-----------------------------------------------
        
        # Copy the frame from the webcam into the sender texture
        glBindTexture(GL_TEXTURE_2D, senderTextureID)
       
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, frame )
      
        # Send texture to Spout
        # Its signature in C++ looks like this: bool SendTexture(GLuint TextureID, GLuint TextureTarget, unsigned int width, unsigned int height, bool bInvert=true, GLuint HostFBO = 0);
        spoutSender.SendTexture(senderTextureID, GL_TEXTURE_2D, spoutSenderWidth, spoutSenderHeight, False, 0)
       
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT  | GL_DEPTH_BUFFER_BIT )
        # reset the drawing perspective
        glLoadIdentity()
       
        # Draw texture to screen
        
        glBegin(GL_QUADS)

        glTexCoord(0,0)        
        glVertex2f(0.0,0)

        glTexCoord(1,0)
        glVertex2f(width,0)

        glTexCoord(1,1)
        glVertex2f(width,height)

        glTexCoord(0.0,1)
        glVertex2f(0,height)

        glEnd()
        

        # update window
        pygame.display.flip()             
      
        # unbind our sender texture
        glBindTexture(GL_TEXTURE_2D, 0)

if __name__ == '__main__':
    main()