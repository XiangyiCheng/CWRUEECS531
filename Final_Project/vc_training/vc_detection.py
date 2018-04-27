import cv2
import numpy as np
import os

def detect_vocal_cord_from_video():
    vocal_cord_cascade=cv2.CascadeClassifier('cascade_14_stage5050.xml')
    video='/home/liutao/intubot/test_videos/Vocal_Cords_in_Action_cut.mp4'
    cap=cv2.VideoCapture(video) # (1) for capturing video
    print 'hello'
    while(cap.isOpened()):
        ret,frame=cap.read()
	
        gray_video=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        vocal_cord=vocal_cord_cascade.detectMultiScale(gray_video,5,60)
        for (x,y,w,h) in vocal_cord:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) # the image to draw, starting point, ending point, color, line thickness.
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
def detect_vocal_cord_from_image():
    vocal_cord_cascade=cv2.CascadeClassifier('cascade_10_stage5050.xml')
    img_path= '/home/pi/intubot/cascade_training/test_pos_img/1.jpg'
    img=cv2.imread(img_path)
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    vocal_cord=vocal_cord_cascade.detectMultiScale(gray_img,5,60)
    for (x,y,w,h) in vocal_cord:
        print 'X coordinate:', x
        print 'Y coordinate:', y
        print 'Width:', w
        print 'Height:',h
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # the image to draw, starting point, ending point, color, line thickness.
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'vocal cord',(x,y-5),font,0.4,(200,255,255),1,cv2.LINE_AA)
    cv2.imshow('vocal_cord_detection',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_vocal_cord_from_folder():
    vocal_cord_cascade=cv2.CascadeClassifier('cascade_14_stage5050.xml')
    total_img=100
    pic_num=1
    pos_detection_num=0
    miss_num=0
    scale_factor=10
    min_neighbors=12
    #if not os.path.exists('result5050_pos'):
        #os.makedirs('result5050_pos')
    for img in os.listdir('test_pos_img'):
        img1= cv2.imread('test_pos_img/'+str(img))
        gray_img=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        vocal_cord=vocal_cord_cascade.detectMultiScale(img1,scale_factor,min_neighbors)
        det_in_each=0
        for (x,y,w,h) in vocal_cord:
            cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
            det_in_each+=1
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img1,'vocal cord',(x,y-5),font,0.4,(200,255,255),1,cv2.LINE_AA)
        if (det_in_each==0):
            miss_num+=1
        pos_detection_num=pos_detection_num+det_in_each
        cv2.imwrite('result5050_pos/'+img,img1)
        print 'Pos_No.', pic_num, 'is done.'
        pic_num+=1
    pic_num=1
    hits_num=0
    neg_detection_num=0
    for img in os.listdir('test_neg_img'):
        img1= cv2.imread('test_neg_img/'+str(img))
        gray_img=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        vocal_cord=vocal_cord_cascade.detectMultiScale(img1,scale_factor,min_neighbors)
        det_in_each=0
        for (x,y,w,h) in vocal_cord:
            cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
            det_in_each+=1
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img1,'vocal cord',(x,y-5),font,0.4,(200,255,255),1,cv2.LINE_AA)
        if (det_in_each==0):
            hits_num+=1
        neg_detection_num=neg_detection_num+det_in_each
        #cv2.imwrite('result5050_neg/'+img,img1)
        print 'Neg_No.', pic_num, 'is done.'
        pic_num+=1
        
    none_correct_in_detected=input('Enter the number of none correct image if detected:')
    multi_correct_in_detected=input('Enter the number of multiple correct cases if detected:')
    total_detection_num=pos_detection_num+neg_detection_num+hits_num+miss_num
    hits_num=hits_num+(total_img-miss_num-none_correct_in_detected)+multi_correct_in_detected
    false_detect_num= neg_detection_num+ (pos_detection_num-(total_img-miss_num))+none_correct_in_detected-multi_correct_in_detected
    miss_num= miss_num+none_correct_in_detected
    TPR = float(hits_num)/ float(total_detection_num)
    FPR = float(false_detect_num)/ float(total_detection_num)
    print 'hits:', hits_num
    print 'missed:',miss_num
    print 'False detection:', false_detect_num
    print 'True Positive Rate (TPR) is:', TPR
    print 'False Positive Rate (FPR) is:', FPR

def detect_vocal_cord_from_image():
    vocal_cord_cascade=cv2.CascadeClassifier('cascade_10_stage5050.xml')
    img_path= '/home/pi/intubot/cascade_training/test_pos_img/1.jpg'
    img=cv2.imread(img_path)
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    vocal_cord=vocal_cord_cascade.detectMultiScale(gray_img,5,60)
    for (x,y,w,h) in vocal_cord:
        print 'X coordinate:', x
        print 'Y coordinate:', y
        print 'Width:', w
        print 'Height:',h
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # the image to draw, starting point, ending point, color, line thickness.
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'vocal cord',(x,y-5),font,0.4,(200,255,255),1,cv2.LINE_AA)
    cv2.imshow('vocal_cord_detection',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_vocal_cord_from_folder_modified():
    vocal_cord_cascade=cv2.CascadeClassifier('cascade_14_stage5050.xml')
    total_img=100
    pic_num=1
    pos_detection_num=0
    miss_num=0
    scale_factor=5
    min_neighbors=33

    for img in os.listdir('test_pos_img'):
        img1= cv2.imread('test_pos_img/'+str(img))
        gray_img=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        vocal_cord=vocal_cord_cascade.detectMultiScale(img1,scale_factor,min_neighbors)
        det_in_each=0
        for (x,y,w,h) in vocal_cord:
            cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
            det_in_each+=1
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img1,'vocal cord',(x,y-5),font,0.4,(200,255,255),1,cv2.LINE_AA)
        if (det_in_each==0):
            miss_num+=1
        pos_detection_num=pos_detection_num+det_in_each
        cv2.imwrite('result5050_pos/'+img,img1)
        print 'Pos_No.', pic_num, 'is done.'
        pic_num+=1
    pic_num=1
    correct_num=0
    for img in os.listdir('test_neg_img'):
        img1= cv2.imread('test_neg_img/'+str(img))
        gray_img=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        vocal_cord=vocal_cord_cascade.detectMultiScale(img1,scale_factor,min_neighbors)
        det_in_each=0
        for (x,y,w,h) in vocal_cord:
            cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
            det_in_each+=1
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img1,'vocal cord',(x,y-5),font,0.4,(200,255,255),1,cv2.LINE_AA)
        if (det_in_each==0):
            correct_num+=1
        #cv2.imwrite('result5050_neg/'+img,img1)
        print 'Neg_No.', pic_num, 'is done.'
        pic_num+=1
        
    none_correct_in_detected=input('Enter the number of none correct image if detected:')
    #multi_correct_in_detected=input('Enter the number of multiple correct cases if detected:')
    hits_num=total_img-miss_num-none_correct_in_detected
    false_detect_num= total_img-correct_num
    miss_num= miss_num+none_correct_in_detected
    TPR = float(hits_num)/ float(total_img)
    FPR = float(false_detect_num)/ float(total_img)
    print 'hits:', hits_num
    print 'missed:',miss_num
    print 'False detection:', false_detect_num
    print 'True Positive Rate (TPR) is:', TPR
    print 'False Positive Rate (FPR) is:', FPR

    
#detect_vocal_cord_from_folder_modified()    
#detect_vocal_cord_from_folder()
#detect_vocal_cord_from_image()
detect_vocal_cord_from_video()
