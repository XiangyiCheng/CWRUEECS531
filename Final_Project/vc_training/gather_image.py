import urllib
#import urllib2
import cv2
import numpy as np
import os

##def store_raw_images():
##    neg_images_link = ''
##    neg_image_urls = urllib2.urlopen(neg_images_link).read().decode()
##    pic_num = 1
##    if not os.path.exists('neg'):
##        os.makedirs('neg')
##    for i in neg_image_urls.split('\n'):
##        try:
##   #print(i)
##        urllib.urlretrieve(i, 'neg/'+str(pic_num)+'.jpg')
##        img = cv2.imread('neg/'+str(pic_num)+'.jpg',cv2.IMREAD_GRAYSCALE)
##        resized_image = cv2.resize(img, (100, 100))
##        cv2.imwrite('neg/'+str(pic_num)+'.jpg',resized_image)
##        pic_num += 1
##        except Exception as e:
##        print('Error',str(pic_num))

def store_raw_image():
    neg_images_link='http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00021265'
    #neg_image_urls=urllib2.urlopen(neg_images_link).read().decode()
    neg_image_urls=urllib.urlopen(neg_images_link).read()
    pic_num=83
    if not os.path.exists('test2'):
        os.makedirs('test2')
   # print neg_image_urls.split('/n')
    for i in neg_image_urls.split('\r\n'):
        try:
            print(i)
            urllib.urlretrieve(i,'test2/'+str(pic_num)+'.jpg')
            #img=cv2.imread('test2/'+str(pic_num)+'.jpg',cv2.IMREAD_GRAYSCALE)
            img=cv2.imread('test2/'+str(pic_num)+'.jpg')
            resized_image=cv2.resize(img,(200,200))
            cv2.imwrite('test2/'+str(pic_num)+'.jpg',resized_image)
            pic_num += 1
        except Exception as e:
            print ('Error',str(pic_num))

def self_image_store():
    neg_images_link='https://hei-heg-hoogeind.dse.nl/images/hhh/boom%20kaetsveld/6%20kale%20linde%20boom%20erik%20van%20asten.jpg'
   # neg_image_urls=urllib2.urlopen(neg_images_link).read().decode()
    if not os.path.exists('neg'):
        os.makedirs('neg')
    try:
         urllib.urlretrieve(neg_images_link,'neg/'+'flower1'+'.jpg')
         img=cv2.imread('neg/'+'flower1'+'.jpg',cv2.IMREAD_GRAYSCALE)
         resized_image=cv2.resize(img,(100,100))
         cv2.imwrite('neg/'+'flower1'+'.jpg',resized_image)
    except Exception as e:
         print ('Error')
         
store_raw_image()
##self_image_store()
