import random
import os
import cv2

def randomly_select_test_image():
    image_list=os.listdir('Closed_vocal_cord_Expansion')
    #image_list=os.listdir('Open_vocal_cord')
    select_num=74
    select_image=random.sample(image_list,select_num)
    test_image_num=27
    if not os.path.exists('test'):
        os.makedirs('test')
    if not os.path.exists('test_resized'):
        os.makedirs('test_resized')
    for i in range (0,select_num):
        try:
            #pre_name= 'Open_vocal_cord/'+ str(select_image[i])
            pre_name='Closed_vocal_cord_Expansion/'+str(select_image[i])
            new_name='test1/'+str(test_image_num)+'.jpg'

            os.rename(pre_name,new_name)
            #raw_test_image=cv2.imread('test/'+str(test_image_num)+'.jpg')
            #resized_image=cv2.resize(raw_test_image,(50,50))
            #cv2.imwrite('test1/'+str(test_image_num)+'.jpg',raw_test_image)
            test_image_num+=1
        except Exception as e:
            print str(e)
            
randomly_select_test_image()
