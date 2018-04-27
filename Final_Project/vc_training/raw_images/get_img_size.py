import cv2
import numpy as np
import os

def get_img_size():
##    total_h=0
##    total_w=0
    min_h=200
    min_w=100
    pic_num=0
    for img in os.listdir('selected_pos_image'):
        img=cv2.imread('selected_pos_image/'+str(img))
        height,width,channels=img.shape
##        total_h= total_h+height
##        total_w=total_w+width
        pic_num+=1
        if (height<min_h):
            min_h=height
        if (width<min_w):
            min_w=width
    print 'minimal height is: ', min_h
    print 'minimal width is: ', min_w
    
##    avg_h= total_h/pic_num
##    avg_w= total_w/pic_num
##    print 'average height is: ', avg_h
##    print 'average width is: ',avg_w

get_img_size()
