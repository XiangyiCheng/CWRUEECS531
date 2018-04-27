import os
import cv2
import numpy as np

def find_uglies():
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path=str(file_type)+'/'+str(img)
                    print ugly
                    ugly = cv2.imread('uglies/'+str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not (np.bitwise_xor(ugly,question).any()):
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))

def find_uglies_modified():
    for img in os.listdir('neg'):
        for ugly in os.listdir('uglies'):
            try:
                # read the images in two folders
                #print type (img)
                question = cv2.imread('neg/'+img)
                ugly = cv2.imread('uglies/'+ugly)
                # compare the nag images with the image in 'uglies', delete them if they are the same.
                if ugly.shape == question.shape and not (np.bitwise_xor(ugly,question).any()):
                    print('neg/'+img)
                    os.remove('neg/'+img)
            except Exception as e:
                print(str(e))

find_uglies_modified()
