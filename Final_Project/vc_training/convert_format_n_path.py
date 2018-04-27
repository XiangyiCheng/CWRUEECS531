import os
import cv2
import numpy as np

def convert_pos():
    pic_num=1
    for img in os.listdir('raw_images/selected_pos_image'):
        try:
            img=cv2.imread('raw_images/selected_pos_image/'+str(img),cv2.IMREAD_GRAYSCALE)
            resized_img=cv2.resize(img,(50,50))
            cv2.imwrite('pos/'+str(pic_num)+'.jpg',resized_img)
            pic_num+=1
        except Exception as e:
            print str(e)

def convert_neg():
    pic_num=1173
    for img in os.listdir('new_neg'):
        try:
            img=cv2.imread('new_neg/'+str(img),cv2.IMREAD_GRAYSCALE)
            resized_img=cv2.resize(img,(50,50))
            cv2.imwrite('neg/'+str(pic_num)+'.jpg',resized_img)
            pic_num+=1
        except Exception as e:
            print str(e)

#convert_neg()
convert_pos()
