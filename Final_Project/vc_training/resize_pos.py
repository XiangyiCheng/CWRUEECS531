import os
import cv2
import numpy as np

def resized_pos_image():
    if not os.path.exists('resized_pos'):
        os.makedirs('resized_pos')
        
    num_pic=611
    #for img in os.listdir('Closed_vocal_cord_Expansion'):
    for img in os.listdir('Open_vocal_cord'):
        try:
            #img = cv2.imread('Closed_vocal_cord_Expansion/'+img,cv2.IMREAD_GRAYSCALE)
            img = cv2.imread('Open_vocal_cord/'+img,cv2.IMREAD_GRAYSCALE)
            resized_img= cv2.resize(img,(50,50))
            cv2.imwrite('resized_pos/'+str(num_pic)+'.jpg',resized_img)
            num_pic+=1

        except Exception as e:
            print str(e)

def resized_test_image():
    if not os.path.exists('resized_pos_test'):
        os.makedirs('resized_pos_test')
        
    num_pic=1
    #for img in os.listdir('Closed_vocal_cord_Expansion'):
    for img in os.listdir('test_pos_img'):
        try:
            #img = cv2.imread('Closed_vocal_cord_Expansion/'+img,cv2.IMREAD_GRAYSCALE)
            img = cv2.imread('test_pos_img/'+img)
            resized_img= cv2.resize(img,(200,200))
            cv2.imwrite('resized_pos_test/'+str(num_pic)+'.jpg',resized_img)
            num_pic+=1

        except Exception as e:
            print str(e)

#resized_pos_image()
resized_test_image()
